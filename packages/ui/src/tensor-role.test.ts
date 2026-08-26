import { describe, expect, it } from "vitest";
import {
  classifyTensorRole,
  colorForTensorRole,
  tensorRoleLabel
} from "./tensor-role";

describe("tensor roles", () => {
  it.each([
    ["model.embed_tokens.weight", "embedding"],
    ["lm_head.weight", "embedding"],
    ["layers.0.self_attn.q_proj.weight", "attention"],
    ["layers.0.mlp.experts.4.down_proj.weight", "mlp"],
    ["layers.0.input_layernorm.weight", "normalization"],
    ["layers.0.self_attn.q_proj.bias", "normalization"],
    ["vision_encoder.conv1.weight", "convolution"],
    ["unknown.weight", "other"]
  ] as const)("classifies %s as %s", (name, role) => {
    expect(classifyTensorRole(name)).toBe(role);
  });

  it("keeps the dtype hue while varying role shades", () => {
    const base = "#6ee7ff";
    expect(colorForTensorRole(base, "attention")).toBe(base);
    expect(colorForTensorRole(base, "embedding")).not.toBe(base);
    expect(colorForTensorRole(base, "mlp")).not.toBe(base);
    expect(tensorRoleLabel("mlp")).toBe("MLP / expert");
  });
});
