import * as vscode from "vscode";
import * as cp from "child_process";

export class AetherStatusBar {
  private item: vscode.StatusBarItem;
  private vibePath: string;

  constructor(vibePath: string) {
    this.vibePath = vibePath;
    this.item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this.item.command = "vibe.aetherStatus";
    this.item.tooltip = "Vibe aether discipline gates";
    this.item.show();
  }

  // 🟡 Fix: expose method instead of letting callers access private field via index notation
  setVibePath(p: string): void {
    this.vibePath = p;
  }

  async refresh(): Promise<void> {
    try {
      const stdout = await runVibeCommand(this.vibePath, ["--aether-status"]);
      const enabled = stdout.includes("enabled");
      this.item.text = enabled ? "$(shield) aether: on" : "$(shield) aether: off";
      this.item.backgroundColor = enabled
        ? undefined
        : new vscode.ThemeColor("statusBarItem.warningBackground");
    } catch {
      this.item.text = "$(shield) aether: ?";
    }
  }

  dispose(): void {
    this.item.dispose();
  }
}

// Use spawn with args array instead of shell interpolation to avoid injection risk.
function runVibeCommand(vibePath: string, args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    let out = "";
    let err = "";
    const proc = cp.spawn(vibePath, args, { env: { ...process.env } });
    proc.stdout.on("data", (d: Buffer) => (out += d.toString()));
    proc.stderr.on("data", (d: Buffer) => (err += d.toString()));
    proc.on("close", (code) => {
      if (code === 0 || out.trim()) resolve(out.trim());
      else reject(new Error(err.trim() || `vibe exited with code ${code ?? "?"}`));
    });
    proc.on("error", reject);
  });
}

export async function runAetherCommand(
  vibePath: string,
  flag: "--enable-aether" | "--disable-aether" | "--aether-status"
): Promise<string> {
  try {
    return await runVibeCommand(vibePath, [flag]);
  } catch (err) {
    const e = err as { message?: string };
    return e.message ?? "Unknown error";
  }
}
