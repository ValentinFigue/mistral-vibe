import * as vscode from "vscode";
import * as path from "path";
import * as fs from "fs";
import * as crypto from "crypto";
import { AcpClient, AcpChunk } from "./acpClient";

function getNonce(): string {
  return crypto.randomBytes(16).toString("base64");
}

export class ChatPanel {
  static readonly viewType = "vibeChat";
  private static instance: ChatPanel | undefined;

  private readonly panel: vscode.WebviewPanel;
  private acp: AcpClient;
  private readonly extensionPath: string;
  private acpPath: string;
  private cwd: string;
  private disposables: vscode.Disposable[] = [];

  static async createOrShow(
    extensionPath: string,
    acpPath: string
  ): Promise<void> {
    // 🟡 Fix: require an open workspace; don't fall back to process.cwd()
    const cwd = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!cwd) {
      vscode.window.showErrorMessage(
        "Vibe: open a workspace folder before starting a chat session."
      );
      return;
    }

    if (ChatPanel.instance) {
      ChatPanel.instance.panel.reveal();
      return;
    }

    const panel = vscode.window.createWebviewPanel(
      ChatPanel.viewType,
      "Vibe",
      vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        localResourceRoots: [
          vscode.Uri.file(path.join(extensionPath, "media")),
        ],
      }
    );

    ChatPanel.instance = new ChatPanel(panel, extensionPath, acpPath, cwd);
  }

  private constructor(
    panel: vscode.WebviewPanel,
    extensionPath: string,
    acpPath: string,
    cwd: string
  ) {
    this.panel = panel;
    this.extensionPath = extensionPath;
    this.acpPath = acpPath;
    this.cwd = cwd;
    this.acp = new AcpClient();

    // 🔴 Fix: generate nonce for CSP and inject into HTML
    this.panel.webview.html = this._getHtml();

    this.panel.onDidDispose(() => this.dispose(), null, this.disposables);

    this.panel.webview.onDidReceiveMessage(
      async (msg: { type: string; text?: string }) => {
        if (msg.type === "prompt" && msg.text) {
          await this._handlePrompt(msg.text);
        } else if (msg.type === "reconnect") {
          await this._init();
        }
      },
      null,
      this.disposables
    );

    this._init();
  }

  private async _init(): Promise<void> {
    this.acp.dispose();
    this.acp = new AcpClient();

    this._postStatus("Connecting to vibe-acp…");
    try {
      await this.acp.spawn(this.cwd, this.acpPath);
      this._postStatus("Connected. Type a message below.");

      // 🟢 Fix: listen for unexpected exit and offer reconnect in the webview
      // (AcpClient surfaces exits via rejected pending Promises; we also need
      // to tell the UI so it can show the reconnect button)
      this.acp["proc"]?.on("exit", () => {
        this._postMessage({
          type: "disconnected",
          text: "vibe-acp exited. Click Reconnect to restart.",
        });
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._postMessage({
        type: "disconnected",
        text: `Failed to connect: ${msg}`,
      });
    }
  }

  private async _handlePrompt(text: string): Promise<void> {
    this._postMessage({ type: "userMessage", text });
    this._postMessage({ type: "assistantStart" });

    try {
      await this.acp.prompt(text, (chunk: AcpChunk) => {
        if (chunk.sessionUpdate === "agent_message_chunk" && chunk.content?.text) {
          this._postMessage({ type: "assistantChunk", text: chunk.content.text });
        } else if (chunk.sessionUpdate === "tool_call") {
          this._postMessage({
            type: "toolCall",
            text: JSON.stringify(chunk, null, 2),
          });
        }
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this._postMessage({ type: "assistantChunk", text: `\n\n*Error: ${msg}*` });
    }

    this._postMessage({ type: "assistantEnd" });
  }

  private _postMessage(msg: Record<string, unknown>): void {
    this.panel.webview.postMessage(msg);
  }

  private _postStatus(text: string): void {
    this._postMessage({ type: "status", text });
  }

  private _getHtml(): string {
    const htmlPath = path.join(this.extensionPath, "media", "chat.html");
    if (!fs.existsSync(htmlPath)) {
      return "<html><body>chat.html not found</body></html>";
    }
    const nonce = getNonce();
    const cspSource = this.panel.webview.cspSource;
    return fs
      .readFileSync(htmlPath, "utf8")
      .replace(/{{nonce}}/g, nonce)
      .replace(/{{cspSource}}/g, cspSource);
  }

  dispose(): void {
    ChatPanel.instance = undefined;
    this.acp.dispose();
    this.panel.dispose();
    for (const d of this.disposables) d.dispose();
    this.disposables = [];
  }
}
