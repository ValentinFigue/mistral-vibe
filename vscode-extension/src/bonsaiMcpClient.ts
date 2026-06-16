import * as cp from "child_process";
import * as readline from "readline";

interface McpToolResult {
  content?: Array<{ type: string; text?: string }>;
  error?: { message: string };
}

// Shared session lifecycle: spawn, MCP handshake, run fn, tear down.
// A fresh process is used per call — bonsai tools are stateless.
async function withBonsaiSession<T>(
  server: "python" | "typescript",
  fn: (
    send: (method: string, params: unknown) => Promise<unknown>,
    notify: (method: string, params: unknown) => void
  ) => Promise<T>
): Promise<T> {
  const [cmd, cmdArgs]: [string, string[]] =
    server === "python"
      ? ["uvx", ["bonsai-python"]]
      : ["npx", ["--yes", "bonsai-ts@latest"]];

  const proc = cp.spawn(cmd, cmdArgs, {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env },
  });

  // 🟡 Fix: guard stdin/stdout before use
  if (!proc.stdin || !proc.stdout) {
    proc.kill();
    throw new Error(`Failed to spawn ${cmd} — stdin/stdout unavailable`);
  }

  const stdin = proc.stdin;
  const rl = readline.createInterface({ input: proc.stdout, crlfDelay: Infinity });
  let nextId = 1;
  const pending = new Map<number, [(v: unknown) => void, (e: Error) => void]>();

  rl.on("line", (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg: Record<string, unknown>;
    try {
      msg = JSON.parse(trimmed) as Record<string, unknown>;
    } catch {
      return;
    }
    const id = msg.id as number | undefined;
    if (id !== undefined) {
      const h = pending.get(id);
      if (h) {
        pending.delete(id);
        if (msg.error) {
          const err = msg.error as { message?: string };
          h[1](new Error(err.message ?? JSON.stringify(msg.error)));
        } else {
          h[0](msg.result ?? {});
        }
      }
    }
  });

  const send = (method: string, params: unknown): Promise<unknown> =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, [resolve, reject]);
      stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });

  const notify = (method: string, params: unknown): void => {
    stdin.write(JSON.stringify({ jsonrpc: "2.0", method, params }) + "\n");
  };

  try {
    await send("initialize", {
      protocolVersion: "2024-11-05",
      capabilities: {},
      clientInfo: { name: "vibe-vscode", version: "0.1.0" },
    });
    notify("notifications/initialized", {});
    return await fn(send, notify);
  } finally {
    rl.close();
    proc.kill();
  }
}

export async function callBonsai(
  server: "python" | "typescript",
  tool: string,
  args: Record<string, unknown>
): Promise<string> {
  return withBonsaiSession(server, async (send) => {
    const result = (await send("tools/call", {
      name: tool,
      arguments: args,
    })) as McpToolResult;

    if (result.error) throw new Error(result.error.message);

    const content = result.content ?? [];
    return content
      .filter((c) => c.type === "text")
      .map((c) => c.text ?? "")
      .join("\n");
  });
}

export async function listBonsaiTools(
  server: "python" | "typescript"
): Promise<Array<{ name: string; description?: string }>> {
  return withBonsaiSession(server, async (send) => {
    const result = (await send("tools/list", {})) as {
      tools?: Array<{ name: string; description?: string }>;
    };
    return result?.tools ?? [];
  });
}
