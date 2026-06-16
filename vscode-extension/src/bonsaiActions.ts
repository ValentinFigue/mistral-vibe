import * as vscode from "vscode";
import { callBonsai } from "./bonsaiMcpClient";

const PYTHON_LANGS = new Set(["python"]);
const TS_LANGS = new Set(["typescript", "typescriptreact", "javascript", "javascriptreact"]);

export class BonsaiCodeActionProvider implements vscode.CodeActionProvider {
  provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range
  ): vscode.CodeAction[] {
    const lang = document.languageId;
    const isPy = PYTHON_LANGS.has(lang);
    const isTs = TS_LANGS.has(lang);
    if (!isPy && !isTs) return [];

    const word = document.getText(document.getWordRangeAtPosition(range.start));
    const actions: vscode.CodeAction[] = [];

    if (word) {
      // Rename symbol
      const rename = new vscode.CodeAction(
        `Bonsai: Rename '${word}'`,
        vscode.CodeActionKind.Refactor
      );
      rename.command = {
        command: isPy ? "vibe.bonsaiRename.python" : "vibe.bonsaiRename.typescript",
        title: "Rename with bonsai",
        arguments: [document.uri.fsPath, word],
      };
      actions.push(rename);

      // Find all references
      const refs = new vscode.CodeAction(
        `Bonsai: Find references to '${word}'`,
        vscode.CodeActionKind.Empty
      );
      refs.command = {
        command: isPy ? "vibe.bonsaiFindRefs.python" : "vibe.bonsaiFindRefs.typescript",
        title: "Find refs with bonsai",
        arguments: [document.uri.fsPath, word],
      };
      actions.push(refs);
    }

    // Move file (available regardless of cursor position)
    const move = new vscode.CodeAction(
      `Bonsai: Move this file`,
      vscode.CodeActionKind.Refactor
    );
    move.command = {
      command: isPy ? "vibe.bonsaiMove.python" : "vibe.bonsaiMove.typescript",
      title: "Move with bonsai",
      arguments: [document.uri.fsPath],
    };
    actions.push(move);

    return actions;
  }
}

export function registerBonsaiCommands(
  context: vscode.ExtensionContext
): void {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "vibe.bonsaiRename.python",
      (filePath: string, symbol: string) =>
        bonsaiRename("python", filePath, symbol)
    ),
    vscode.commands.registerCommand(
      "vibe.bonsaiRename.typescript",
      (filePath: string, symbol: string) =>
        bonsaiRename("typescript", filePath, symbol)
    ),
    vscode.commands.registerCommand(
      "vibe.bonsaiFindRefs.python",
      (filePath: string, symbol: string) =>
        bonsaiFindRefs("python", filePath, symbol)
    ),
    vscode.commands.registerCommand(
      "vibe.bonsaiFindRefs.typescript",
      (filePath: string, symbol: string) =>
        bonsaiFindRefs("typescript", filePath, symbol)
    ),
    vscode.commands.registerCommand(
      "vibe.bonsaiMove.python",
      (filePath: string) => bonsaiMove("python", filePath)
    ),
    vscode.commands.registerCommand(
      "vibe.bonsaiMove.typescript",
      (filePath: string) => bonsaiMove("typescript", filePath)
    )
  );
}

async function bonsaiRename(
  server: "python" | "typescript",
  filePath: string,
  symbol: string
): Promise<void> {
  const newName = await vscode.window.showInputBox({
    prompt: `Rename '${symbol}' to:`,
    value: symbol,
    validateInput: (v) => (v.trim() ? null : "Name cannot be empty"),
  });
  if (!newName || newName === symbol) return;

  const toolName = server === "python" ? "pyrename" : "tsrename";
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Bonsai: renaming '${symbol}' → '${newName}'…` },
    async () => {
      try {
        const output = await callBonsai(server, toolName, {
          file_path: filePath,
          symbol,
          new_name: newName,
        });
        vscode.window.showInformationMessage(
          `Bonsai rename complete:\n${output.slice(0, 200)}`
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Bonsai rename failed: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    }
  );
}

async function bonsaiFindRefs(
  server: "python" | "typescript",
  filePath: string,
  symbol: string
): Promise<void> {
  const toolName = server === "python" ? "pyfindrefs" : "tsfindrefs";
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Bonsai: finding references to '${symbol}'…` },
    async () => {
      try {
        const output = await callBonsai(server, toolName, {
          file_path: filePath,
          symbol,
        });
        const panel = vscode.window.createOutputChannel("Bonsai References");
        panel.clear();
        panel.appendLine(`References to '${symbol}':\n`);
        panel.appendLine(output);
        panel.show(true);
      } catch (err) {
        vscode.window.showErrorMessage(
          `Bonsai find refs failed: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    }
  );
}

async function bonsaiMove(
  server: "python" | "typescript",
  filePath: string
): Promise<void> {
  const dest = await vscode.window.showInputBox({
    prompt: "Move file to (absolute or relative path):",
    value: filePath,
  });
  if (!dest || dest === filePath) return;

  const toolName = server === "python" ? "pymove" : "tsmove";
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: `Bonsai: moving file…` },
    async () => {
      try {
        const output = await callBonsai(server, toolName, {
          source: filePath,
          destination: dest,
        });
        vscode.window.showInformationMessage(
          `Bonsai move complete:\n${output.slice(0, 200)}`
        );
      } catch (err) {
        vscode.window.showErrorMessage(
          `Bonsai move failed: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    }
  );
}
