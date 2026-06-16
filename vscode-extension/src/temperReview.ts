import * as vscode from "vscode";
import * as cp from "child_process";

const diagnostics = vscode.languages.createDiagnosticCollection("temper");

// 🟢 Fix: reject separator rows (|---|---|---|) and header rows by requiring
// cell content to have at least one non-dash, non-space, non-pipe character.
const CELL_HAS_CONTENT = /[^\s|:-]/;

function parseFindings(
  output: string
): Array<{ severity: vscode.DiagnosticSeverity; file: string; finding: string }> {
  const results: Array<{
    severity: vscode.DiagnosticSeverity;
    file: string;
    finding: string;
  }> = [];

  // Row format: | # | Dimension | Severity | File | Finding | Recommendation |
  const tableRow = /^\|[^|]*\|[^|]*\|([^|]+)\|([^|]+)\|([^|]+)\|/;

  for (const line of output.split("\n")) {
    const m = tableRow.exec(line.trim());
    if (!m) continue;

    const severityCell = m[1].trim();
    const fileCell = m[2].trim();
    const findingCell = m[3].trim();

    // Skip header and separator rows
    if (
      !CELL_HAS_CONTENT.test(severityCell) ||
      !CELL_HAS_CONTENT.test(findingCell) ||
      findingCell === "Finding" ||
      findingCell === "Severity"
    ) {
      continue;
    }

    let severity: vscode.DiagnosticSeverity;
    if (severityCell.includes("🔴") || severityCell.toLowerCase().includes("blocker")) {
      severity = vscode.DiagnosticSeverity.Error;
    } else if (severityCell.includes("🟡") || severityCell.toLowerCase().includes("significant")) {
      severity = vscode.DiagnosticSeverity.Warning;
    } else {
      severity = vscode.DiagnosticSeverity.Information;
    }

    results.push({ severity, file: fileCell, finding: findingCell });
  }

  return results;
}

// 🟡 Fix: use spawn with args array instead of execAsync shell interpolation.
function spawnVibe(vibePath: string, args: string[], cwd: string, timeoutMs: number): Promise<string> {
  return new Promise((resolve, reject) => {
    let out = "";
    let err = "";
    const proc = cp.spawn(vibePath, args, { cwd, env: { ...process.env } });

    const timer = setTimeout(() => {
      proc.kill();
      // Resolve with whatever we got so far rather than dropping it
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

export async function runTemperReview(vibePath: string): Promise<void> {
  const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
  // 🟡 Fix: don't fall back to process.cwd() — show a clear error
  if (!cwd) {
    vscode.window.showErrorMessage("Vibe: no workspace folder open. Open a folder to use temper review.");
    return;
  }

  // Check that something is staged
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
      title: "Vibe: reviewing staged diff (temper)…",
      cancellable: false,
    },
    async () => {
      let output = "";
      try {
        output = await spawnVibe(
          vibePath,
          ["--prompt", "/temper", "--output", "text", "--auto-approve", "--trust"],
          cwd,
          120_000
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Vibe temper failed: ${err instanceof Error ? err.message : String(err)}`
        );
        return;
      }

      const channel = vscode.window.createOutputChannel("Vibe: Temper Review");
      channel.clear();
      channel.appendLine(output);
      channel.show(true);

      diagnostics.clear();
      const findings = parseFindings(output);

      if (findings.length === 0) {
        vscode.window.showInformationMessage("Vibe temper: no findings. Staged diff looks clean.");
        return;
      }

      const byFile = new Map<string, vscode.Diagnostic[]>();
      for (const { severity, file, finding } of findings) {
        const uri = file && file !== "—" && file !== "-"
          ? vscode.Uri.file(file.startsWith("/") ? file : `${cwd}/${file}`)
          : vscode.Uri.file(cwd);

        const diag = new vscode.Diagnostic(
          new vscode.Range(0, 0, 0, 0),
          `temper: ${finding}`,
          severity
        );
        diag.source = "temper";

        const key = uri.toString();
        if (!byFile.has(key)) byFile.set(key, []);
        byFile.get(key)!.push(diag);
      }

      for (const [uriStr, diags] of byFile) {
        diagnostics.set(vscode.Uri.parse(uriStr), diags);
      }

      const blockers = findings.filter(
        (f) => f.severity === vscode.DiagnosticSeverity.Error
      ).length;

      if (blockers > 0) {
        vscode.window.showWarningMessage(
          `Vibe temper: ${blockers} blocker(s) found. Fix before committing.`
        );
      } else {
        vscode.window.showInformationMessage(
          `Vibe temper: ${findings.length} finding(s), no blockers. Safe to commit.`
        );
      }
    }
  );
}

export function getTemperDiagnostics(): vscode.DiagnosticCollection {
  return diagnostics;
}
