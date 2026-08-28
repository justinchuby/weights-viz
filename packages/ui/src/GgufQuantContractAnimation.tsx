import { Binary, Boxes, Calculator, ChevronRight, Cpu, Layers3 } from "lucide-react";
import type { ReactNode } from "react";
import {
  AnimationControls,
  AnimationStepCopy,
  type AnimationStep,
  useAnimationPlayer
} from "./DtypeAnimationPlayer";
import { ggufQuantContract } from "./gguf-quant-contracts";
import { ggufStorageLayout } from "./gguf-storage-layouts";

const STEPS: readonly AnimationStep[] = [
  {
    label: "Fix the ABI scope",
    detail:
      "The dtype—not the file—fixes the number of weights, stored bytes, and metadata-sharing groups."
  },
  {
    label: "Derive the metadata",
    detail:
      "Compute the exact block or group parameters, then round and pack them in their declared storage types."
  },
  {
    label: "Choose the codes",
    detail:
      "Map each source value or vector to the format’s real integer, ternary, floating, or codebook representation."
  },
  {
    label: "Write the physical record",
    detail:
      "Fields appear in a fixed order; split nibbles, bit-planes, and embedded metadata are part of the ABI."
  },
  {
    label: "Decode inside the kernel",
    detail:
      "Reassemble one code and its exact metadata scopes, then reconstruct only the values needed by the fused operation."
  }
];

export function GgufQuantContractAnimation({ dtype }: { dtype: string }) {
  const contract = ggufQuantContract(dtype);
  const fields = ggufStorageLayout(dtype);
  const player = useAnimationPlayer(STEPS.length);
  const { step } = player;

  if (!contract || !fields) return null;

  const bitsPerWeight = (contract.bytes * 8) / contract.values;

  return (
    <section className="wv-contract-demo">
      <header>
        <div>
          <span>EXACT BLOCK CONTRACT</span>
          <h3>
            <Layers3 aria-hidden="true" />
            Open the {dtype} record
          </h3>
          <p>
            {contract.family}. Every field, group boundary, code rule, and
            reconstruction step below belongs to this dtype’s fixed GGML ABI.
          </p>
        </div>
        <div className="wv-contract-ratio">
          <small>effective storage</small>
          <strong>{formatDecimal(bitsPerWeight)} bpw</strong>
        </div>
      </header>

      <div className="wv-contract-player">
        <AnimationStepCopy
          step={step}
          steps={STEPS}
          announce={!player.playing}
        />

        <div className={`wv-contract-body step-${step}`}>
          <div className="wv-contract-facts">
            <Fact label="Block" value={`${contract.values} weights`} />
            <Fact label="Record" value={`${contract.bytes} bytes`} />
            <Fact
              label="Grouping"
              value={
                contract.groups.count === 1
                  ? contract.groups.label
                  : `${contract.groups.count} × ${contract.groups.values}`
              }
            />
            <Fact label="Size rule" value="fixed, not configurable" />
          </div>

          <div className="wv-contract-groups" aria-label={`${dtype} group map`}>
            {Array.from({ length: contract.groups.count }, (_, group) => (
              <span key={group}>
                <b>{contract.groups.count === 1 ? "block" : `g${group}`}</b>
                <small>
                  {group * contract.groups.values}…
                  {(group + 1) * contract.groups.values - 1}
                </small>
              </span>
            ))}
          </div>

          <div className="wv-contract-panels">
            <ContractPanel
              icon={<Calculator aria-hidden="true" />}
              title="Metadata derivation"
              items={contract.metadata}
              active={step === 1}
              revealed={step >= 1}
            />
            <ChevronRight aria-hidden="true" />
            <ContractPanel
              icon={<Binary aria-hidden="true" />}
              title="Code contract"
              items={contract.codes}
              active={step === 2}
              revealed={step >= 2}
            />
          </div>

          <div className={`wv-contract-layout${step === 3 ? " active" : ""}`}>
            <header>
              <Boxes aria-hidden="true" />
              <span>
                <strong>Physical {dtype} record</strong>
                <small>exact field order · {contract.bytes} bytes total</small>
              </span>
            </header>
            <div className="wv-contract-field-strip">
              {fields.map((field) => (
                <span key={field.name} style={{ flexGrow: field.bytes }}>
                  <strong>{field.name}</strong>
                  <small>
                    {field.type}[{field.count}] · {field.bytes} B
                  </small>
                </span>
              ))}
            </div>
            <ol>
              {contract.packing.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ol>
          </div>

          <div className={`wv-contract-decode${step === 4 ? " active" : ""}`}>
            <div>
              <Cpu aria-hidden="true" />
              <span>
                <small>exact reconstruction</small>
                <code>{contract.decode}</code>
              </span>
            </div>
            <ChevronRight aria-hidden="true" />
            <p>{contract.runtime}</p>
            <section className="wv-contract-symbols">
              <header>
                <strong>Every symbol, traced</strong>
                <small>formula name → meaning → exact source</small>
              </header>
              <dl>
                {contract.symbols.map((item, index) => (
                  <div key={`${item.symbol}-${index}`}>
                    <dt>
                      <code>{item.symbol}</code>
                      <span>{item.meaning}</span>
                    </dt>
                    <dd>{item.source}</dd>
                  </div>
                ))}
              </dl>
            </section>
          </div>
        </div>
      </div>

      <AnimationControls
        steps={STEPS}
        step={step}
        playing={player.playing}
        onPrevious={player.previous}
        onNext={player.next}
        onSelect={player.selectStep}
        onToggle={player.togglePlaying}
        onRestart={player.restart}
      />
    </section>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <span>
      <small>{label}</small>
      <strong>{value}</strong>
    </span>
  );
}

function ContractPanel({
  icon,
  title,
  items,
  active,
  revealed
}: {
  icon: ReactNode;
  title: string;
  items: readonly string[];
  active: boolean;
  revealed: boolean;
}) {
  return (
    <article
      className={`${active ? "active" : ""}${revealed ? " revealed" : ""}`}
    >
      <header>
        {icon}
        <strong>{title}</strong>
      </header>
      <ol>
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ol>
    </article>
  );
}

function formatDecimal(value: number): string {
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
