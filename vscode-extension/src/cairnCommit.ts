import * as vscode from "vscode";
import * as cp from "child_process";

// 🟢 Fix: take the LAST fenced code block — vibe may echo the diff before the message.
function extractCommitMessage(output: string): string | null {
  const fencePattern = /```[^\n]*\n([\s\S]*?)```/g;
  let lastMatch: RegExpExecArray | null = null;
  let m: RegExpExecArray | null;
  while ((m = fencePattern.exec(output)) !== null) {
    lastMatch = m;
  }
  if (lastMatch) return lastMatch[1].trim();

  // Fallback: first line matching a conventional commit type
  const conventionalMatch =
    /^(feat|fix|docs|style|refactor|test|chore|build|ci|perf)(\([^)]+\))?:\s*.+/m.exec(output);
  if (conventionalMatch) {
    return output.slice(output.indexOf(conventionalMatch[0])).trim();
  }

  return null;
}

// 🟡 Fix: use spawn with args array instead of execAsync shell interpolation.
function spawnVibe(vibePath: string, args: string[], cwd: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    let out = "";
    let err = "";
    const proc = cp.spawn(vibePath, args, { cwd, env: { ...process.env } });

    const timer = setTimeout(() => {
      proc.kill();
      if (out.trim()) resolve(out);
      else reject(new Error(`vibe timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    proc.stdout.on("data", (d: Buffer) => (out += d.toString()));
    proc.stderr.on("data", (d: Buffer) => (err += d.toString()));
    proc.on("close", () => {
      clearTimeout(timer);
      if (out.trim()) resolve(out);
      else reject(new Error(err.trim() || "vibe produced no output"));
    });
    proc.on("error", (e) => { clearTimeout(timer); reject(e); });
  });
}

export async function generateCairnCommit(vibePath: string): Promise<void> {
  const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  // 🟡 Fix: don't fall back to process.cwd()
  if (!cwd) {
    vscode.window.showErrorMessage("Vibe: no workspace folder open.");
    return;
  }

  // Check staged changes
  try {
    const stat = await spawnVibe("git", ["diff", "--cached", "--stat"], cwd, 10_000);
    if (!stat.trim()) {
      vscode.window.showInformationMessage("Vibe: nothing staged. Run `git add` first.");
      return;
    }
  } catch {
    vscode.window.showErrorMessage("Vibe: failed to run git — is this a git repo?");
    return;
  }

  await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title: "Vibe: generating commit message (cairn)…",
      cancellable: false,
    },
    async () => {
      let output = "";
      try {
        output = await spawnVibe(
          vibePath,
          ["--prompt", "/cairn-commit", "--output", "text", "--auto-approve", "--trust"],
          cwd,
          60_000
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Vibe cairn failed: ${err instanceof Error ? err.message : String(err)}`
        );
        return;
      }

      const message = extractCommitMessage(output);
      if (!message) {
        vscode.window.showWarningMessage(
          "Vibe: could not extract a commit message from the response."
        );
        return;
      }

      // Populate the SCM input box via the Git extension API
      const gitExtension = vscode.extensions.getExtension("vscode.git")?.exports;
      const api = gitExtension?.getAPI?.(1);
      const repo = api?.repositories?.[0];

      if (repo) {
        repo.inputBox.value = message;
        vscode.window.showInformationMessage(
          "Vibe: commit message generated and placed in the Source Control input box."
        );
      } else {
        // Fallback: show in input box for manual copy
        await vscode.window.showInputBox({
          prompt: "Generated commit message (copy and paste):",
          value: message,
        });
      }
    }
  );
}
