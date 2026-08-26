export type TensorRole =
  | "attention"
  | "convolution"
  | "embedding"
  | "mlp"
  | "normalization"
  | "other";

const ROLE_LIGHTNESS: Record<TensorRole, number> = {
  attention: 0,
  convolution: 0.08,
  embedding: 0.15,
  mlp: -0.1,
  normalization: -0.18,
  other: -0.04
};

const ROLE_LABELS: Record<TensorRole, string> = {
  attention: "Attention",
  convolution: "Convolution",
  embedding: "Embedding / output",
  mlp: "MLP / expert",
  normalization: "Normalization / bias",
  other: "Other"
};

export function classifyTensorRole(name: string): TensorRole {
  const normalized = name.toLowerCase();
  if (
    /(^|[./_])(norm|layer_norm|layernorm|rms_norm|rmsnorm)([./_]|$)/.test(
      normalized
    ) ||
    /(^|[./_])bias$/.test(normalized)
  ) {
    return "normalization";
  }
  if (
    /(^|[./_])(lm_head|embed|embeddings?|embed_tokens?|word_embeddings?|wte)([./_]|$)/.test(
      normalized
    )
  ) {
    return "embedding";
  }
  if (
    /(^|[./_])(attention|attn|self_attn|linear_attn|q_proj|k_proj|v_proj|o_proj|qkv|in_proj_qkv)([./_]|$)/.test(
      normalized
    )
  ) {
    return "attention";
  }
  if (
    /(^|[./_])(mlp|ffn|feed_forward|experts?|router|gate_proj|up_proj|down_proj|fc\d*)([./_]|$)/.test(
      normalized
    )
  ) {
    return "mlp";
  }
  if (/(^|[./_])(conv\d*|convolution)([./_]|$)/.test(normalized)) {
    return "convolution";
  }
  return "other";
}

export function tensorRoleLabel(role: TensorRole): string {
  return ROLE_LABELS[role];
}

export function colorForTensorRole(baseColor: string, role: TensorRole): string {
  const match = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(baseColor);
  if (!match) return baseColor;
  const amount = ROLE_LIGHTNESS[role];
  if (amount === 0) return baseColor;
  const target = amount > 0 ? 255 : 0;
  const ratio = Math.abs(amount);
  const channels = match.slice(1).map((channel) => {
    const value = Number.parseInt(channel, 16);
    return Math.round(value + (target - value) * ratio)
      .toString(16)
      .padStart(2, "0");
  });
  return `#${channels.join("")}`;
}
