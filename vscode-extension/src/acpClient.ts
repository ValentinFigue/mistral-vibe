import * as cp from "child_process";
import * as readline from "readline";

const ACP_TIMEOUT_MS = 30_000;

export interface AcpChunk {
  sessionUpdate: string;
  content?: { type: string; text: string };
  [key: string]: unknown;
}

type PendingResolve = (value: unknown) => void;
type PendingReject = (reason: Error) => void;

export class AcpClient {
  private proc: cp.ChildProcess | null = null;
  private rl: readline.Interface | null = null;
  private nextId = 1;
  private pending = new Map<number, [PendingResolve, PendingReject]>();
  private updateHandler: ((chunk: AcpChunk) => void) | null = null;
  private sessionId: string | null = null;

  async spawn(cwd: string, acpPath: string): Promise<void> {
    this.proc = cp.spawn(acpPath, [], {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env },
    });

    this.proc.stderr?.on("data", () => {
      // Suppress stderr — vibe-acp logs there; errors surface via JSON-RPC
    });

    this.rl = readline.createInterface({
      input: this.proc.stdout!,
      crlfDelay: Infinity,
    });

    this.rl.on("line", (line) => {
      const trimmed = line.trim();
      if (!trimmed) return;
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(trimmed) as Record<string, unknown>;
      } catch {
        return;
      }
      this._dispatch(msg);
    });

    this.proc.on("exit", () => {
      for (const [, [, reject]] of this.pending) {
        reject(new Error("vibe-acp process exited unexpectedly"));
      }
      this.pending.clear();
    });

    // 1. initialize
    await this._send("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: { terminal: true },
      clientInfo: { name: "vibe-vscode", version: "0.1.0" },
    });

    // 2. session/new — pass bonsai MCP servers so vibe wires them automatically
    const result = await this._send("session/new", {
      cwd,
      additionalDirectories: [],
      mcpServers: [
        { name: "bonsai-python", command: "uvx", args: ["bonsai-python"], env: [] },
        { name: "bonsai-ts", command: "npx", args: ["--yes", "bonsai-ts@latest"], env: [] },
      ],
    });

    const res = result as Record<string, unknown>;
    this.sessionId = res.sessionId as string;
  }

  async prompt(
    text: string,
    onChunk: (chunk: AcpChunk) => void
  ): Promise<void> {
    if (!this.sessionId) throw new Error("ACP session not started");
    this.updateHandler = onChunk;
    try {
      await this._send("session/prompt", {
        sessionId: this.sessionId,
        prompt: [{ type: "text", text }],
      });
    } finally {
      this.updateHandler = null;
    }
  }

  dispose(): void {
    this.rl?.close();
    this.proc?.kill();
    this.proc = null;
    this.rl = null;
    this.sessionId = null;
  }

  private _dispatch(msg: Record<string, unknown>): void {
    // Notification (session/update) — no id or id matches nothing pending
    if (msg.method === "session/update" && this.updateHandler) {
      const params = msg.params as Record<string, unknown> | undefined;
      const update = params?.update as AcpChunk | undefined;
      if (update) this.updateHandler(update);
      return;
    }

    // Response to a pending request
    const id = msg.id as number | undefined;
    if (id !== undefined) {
      const handlers = this.pending.get(id);
      if (!handlers) return;
      this.pending.delete(id);
      const [resolve, reject] = handlers;
      if (msg.error) {
        reject(new Error(JSON.stringify(msg.error)));
      } else {
        resolve(msg.result ?? {});
      }
    }
  }

  private _send(method: string, params: unknown): Promise<unknown> {
    return new Promise((resolve, reject) => {
      // 🔴 Fix: guard stdin before enqueuing — otherwise the Promise hangs forever
      if (!this.proc?.stdin) {
        reject(new Error(`Cannot send '${method}': vibe-acp is not running`));
        return;
      }

      const id = this.nextId++;

      // 🟡 Fix: timeout so initialize/session/new can't hang indefinitely
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(
          new Error(`ACP request '${method}' timed out after ${ACP_TIMEOUT_MS}ms`)
        );
      }, ACP_TIMEOUT_MS);

      this.pending.set(id, [
        (v) => { clearTimeout(timer); resolve(v); },
        (e) => { clearTimeout(timer); reject(e); },
      ]);

      this.proc.stdin.write(
        JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n"
      );
    });
  }
}
