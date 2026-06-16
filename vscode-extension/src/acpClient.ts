import * as cp from "child_process";
import * as os from "os";
import * as path from "path";
import * as readline from "readline";

// Bonsai is a local Claude Code plugin — not on PyPI/npm under these names.
// The plugin lives in ~/.claude/plugins/marketplaces/bonsai.
const BONSAI_DIR = path.join(os.homedir(), ".claude", "plugins", "marketplaces", "bonsai");

const ACP_TIMEOUT_MS = 30_000;
const PROMPT_TIMEOUT_MS = 5 * 60 * 1000; // 5 min — streaming response arrives after all chunks

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

    // 2. session/new — bonsai is a local Claude Code plugin, not on PyPI/npm.
    // uvx --from <dir> installs from the local pyproject.toml, bypassing the
    // broken public PyPI release. bonsai-ts is invoked via node directly since
    // it has no npm release.
    const result = await this._send("session/new", {
      cwd,
      additionalDirectories: [],
      mcpServers: [
        {
          name: "bonsai-python",
          command: "uvx",
          args: ["--from", BONSAI_DIR, "bonsai-python"],
          env: [],
        },
        {
          name: "bonsai-ts",
          command: "node",
          args: [path.join(BONSAI_DIR, "ts", "bin", "bonsai-ts.js")],
          env: [],
        },
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
      }, PROMPT_TIMEOUT_MS);
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

  private _send(method: string, params: unknown, timeoutMs = ACP_TIMEOUT_MS): Promise<unknown> {
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
          new Error(`ACP request '${method}' timed out after ${timeoutMs}ms`)
        );
      }, timeoutMs);

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
