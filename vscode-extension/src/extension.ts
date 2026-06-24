import * as vscode from "vscode";
import { ChatPanel } from "./chatPanel";
import { AetherStatusBar, runAetherCommand } from "./statusBar";
import { BonsaiCodeActionProvider, registerBonsaiCommands } from "./bonsaiActions";
import { runTemperReview, getTemperDiagnostics } from "./temperReview";
import { generateCairnCommit } from "./cairnCommit";

function getConfig(): { vibePath: string; acpPath: string } {
  const cfg = vscode.workspace.getConfiguration("vibe");
  return {
    vibePath: cfg.get<string>("executablePath") ?? "vibe",
    acpPath: cfg.get<string>("acpExecutablePath") ?? "vibe-acp",
  };
}

export function activate(context: vscode.ExtensionContext): void {
  const { vibePath, acpPath } = getConfig();

  // --- Aether status bar ---
  const statusBar = new AetherStatusBar(vibePath);
  context.subscriptions.push(statusBar);
  statusBar.refresh();

  // --- Commands ---

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.openChat", () => {
      const { acpPath: acp } = getConfig();
      ChatPanel.createOrShow(context.extensionPath, acp).catch((err) => {
        vscode.window.showErrorMessage(
          `Vibe: failed to open chat — ${err instanceof Error ? err.message : String(err)}`
        );
      });
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.aetherStatus", async () => {
      const { vibePath: vp } = getConfig();
      const output = await runAetherCommand(vp, "--aether-status");
      const choice = await vscode.window.showQuickPick(
        [
          { label: "$(shield) Enable aether", id: "enable" },
          { label: "$(circle-slash) Disable aether", id: "disable" },
        ],
        { title: `Aether status:\n${output}`, placeHolder: "Take an action…" }
      );
      if (choice?.id === "enable") {
        vscode.commands.executeCommand("vibe.enableAether");
      } else if (choice?.id === "disable") {
        vscode.commands.executeCommand("vibe.disableAether");
      }
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.enableAether", async () => {
      const { vibePath: vp } = getConfig();
      const output = await runAetherCommand(vp, "--enable-aether");
      vscode.window.showInformationMessage(`Vibe: ${output}`);
      await statusBar.refresh();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.disableAether", async () => {
      const { vibePath: vp } = getConfig();
      const output = await runAetherCommand(vp, "--disable-aether");
      vscode.window.showInformationMessage(`Vibe: ${output}`);
      await statusBar.refresh();
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.temperReview", () => {
      const { vibePath: vp } = getConfig();
      return runTemperReview(vp);
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("vibe.cairnCommit", () => {
      const { vibePath: vp } = getConfig();
      return generateCairnCommit(vp);
    })
  );

  // --- Bonsai code actions ---
  registerBonsaiCommands(context);

  const bonsaiProvider = new BonsaiCodeActionProvider();
  context.subscriptions.push(
    vscode.languages.registerCodeActionsProvider(
      [
        { language: "python" },
        { language: "typescript" },
        { language: "typescriptreact" },
        { language: "javascript" },
        { language: "javascriptreact" },
      ],
      bonsaiProvider,
      {
        providedCodeActionKinds: [
          vscode.CodeActionKind.Refactor,
          vscode.CodeActionKind.Empty,
        ],
      }
    )
  );

  // --- Diagnostics collection (temper) ---
  context.subscriptions.push(getTemperDiagnostics());

  // Refresh aether status on config change
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("vibe")) {
        const { vibePath: vp } = getConfig();
        statusBar.setVibePath(vp);
        statusBar.refresh();
      }
    })
  );
}

export function deactivate(): void {
  // Disposables are collected via context.subscriptions — nothing extra needed.
}
