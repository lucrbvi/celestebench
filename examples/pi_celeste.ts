// Bridges Pi to one CelesteBench MCP episode over its HTTP JSON-RPC endpoint.
// The Python harness owns the server (lifecycle, timeout, recording) and passes
// its URL, bearer token and frame cap through the environment.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  const url = process.env.CELESTEBENCH_MCP_URL!;
  const token = process.env.CELESTEBENCH_MCP_TOKEN!;
  const maxFrames = Number(process.env.CELESTEBENCH_MAX_FRAMES || "30");

  const call = async (name: string, args: unknown) => {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Accept: "application/json, text/event-stream",
      },
      body: JSON.stringify({
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: { name, arguments: args },
      }),
    });
    const text = await response.text();
    const body = text.startsWith("data:")
      ? JSON.parse(text.split("\n").find((line) => line.startsWith("data:"))!.slice(5))
      : JSON.parse(text);
    if (body.error) throw new Error(JSON.stringify(body.error));
    if (body.result?.isError) {
      const text = (body.result.content || [])
        .filter((block: any) => block.type === "text")
        .map((block: any) => block.text)
        .join("\n");
      throw new Error(text || "the game rejected this call");
    }
    return (body.result?.content || []).map((block: any) =>
      block.type === "image"
        ? { type: "image" as const, data: block.data, mimeType: block.mimeType }
        : { type: "text" as const, text: block.text },
    );
  };

  const action = Type.Object({
    buttons: Type.Optional(Type.Integer({ minimum: 0, maximum: 63 })),
    action: Type.Optional(Type.String({ description: 'Use "wait" to advance with no buttons' })),
    frames: Type.Integer({ minimum: 1, maximum: maxFrames }),
  }, { additionalProperties: false });

  pi.registerTool({
    name: "observe",
    label: "observe",
    description: "Observe the current game decision without advancing it.",
    parameters: Type.Object({}),
    async execute() {
      return { content: await call("observe", {}), details: {} };
    },
  });

  pi.registerTool({
    name: "play",
    label: "play",
    description: "Apply a non-empty ordered action batch, then observe the result.",
    parameters: Type.Object({
      actions: Type.Array(action, { minItems: 1 }),
    }),
    async execute(_toolCallId, params) {
      return { content: await call("play", params), details: {} };
    },
  });
}
