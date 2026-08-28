import { Binary, Boxes, Calculator, Cpu, Layers3 } from "lucide-react";
import {
  AnimationControls,
  AnimationStepCopy,
  type AnimationStep,
  useAnimationPlayer
} from "./DtypeAnimationPlayer";
import {
  ggufQuantContract,
  type GgufQuantWorkedStage
} from "./gguf-quant-contracts";

const STEPS: readonly AnimationStep[] = [
  {
    label: "Select a source weight",
    detail: "Follow one concrete weight inside one fixed metadata group."
  },
  {
    label: "Select metadata and code",
    detail: "Choose the exact shared parameters, code, and runtime table entry needed by that weight."
  },
  {
    label: "Extract record bits",
    detail: "Locate the selected data in named fields, indices, and packed bit slices."
  },
  {
    label: "Reconstruct the value",
    detail: "Apply the dtype’s decode rule to the selected weight; no new storage terms are introduced here."
  }
];

export function GgufQuantContractAnimation({ dtype }: { dtype: string }) {
  const contract = ggufQuantContract(dtype);
  const player = useAnimationPlayer(STEPS.length);
  const { step } = player;

  if (!contract) return null;

  const bitsPerWeight = (contract.bytes * 8) / contract.values;
  const { selection, stages } = contract.worked;

  return (
    <section className="wv-contract-demo">
      <header>
        <div>
          <span>WORKED RECORD TRANSFORMATION</span>
          <h3>
            <Layers3 aria-hidden="true" />
            Trace {dtype} weight {selection.weight}
          </h3>
          <p>
            One concrete weight position from source selection through packed storage to
            reconstruction. All field names and formula terms are defined in the
            Storage contract above.
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
              label="Group"
              value={`group ${selection.group} = weights ${
                selection.group * contract.groups.values
              }…${(selection.group + 1) * contract.groups.values - 1}`}
            />
            <Fact
              label="Position in group"
              value={`${selection.position} → i = ${selection.group}×${
                contract.groups.values
              } + ${selection.position} = ${selection.weight}`}
            />
          </div>

          <div className="wv-contract-groups" aria-label={`${dtype} group map`}>
            {Array.from({ length: contract.groups.count }, (_, group) => (
              <span
                className={group === selection.group ? "selected" : ""}
                key={group}
              >
                <b>{`group ${group}`}</b>
                <small>
                  {group * contract.groups.values}…
                  {(group + 1) * contract.groups.values - 1}
                </small>
              </span>
            ))}
          </div>

          <ol className="wv-contract-worked" aria-label={`${dtype} worked transformation`}>
            {stages.map((stage, index) => (
              <WorkedStage
                active={step === index}
                complete={step >= index}
                key={stage.kind}
                number={index + 1}
                stage={stage}
              />
            ))}
          </ol>
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

function WorkedStage({
  active,
  complete,
  number,
  stage
}: {
  active: boolean;
  complete: boolean;
  number: number;
  stage: GgufQuantWorkedStage;
}) {
  return (
    <li className={`${complete ? "complete" : ""}${active ? " active" : ""}`}>
      <header>
        <span>{number}</span>
        {stageIcon(stage.kind)}
        <strong>{stage.title}</strong>
        <small>{stage.kind}</small>
      </header>
      <p>{stage.detail}</p>
      {stage.kind === "storage" && (
        <div className="wv-contract-accesses">
          {stage.accesses.map((item) => (
            <span key={`${item.field}-${item.index}-${item.bits}`}>
              <code>
                {item.field}[{item.index}] · {item.bits}
              </code>
              <small>{item.action}</small>
            </span>
          ))}
        </div>
      )}
      {stage.symbols.length > 0 && (
        <footer>
          <small>terms defined above</small>
          <div>
            {stage.symbols.map((symbol) => (
              <code key={symbol}>{symbol}</code>
            ))}
          </div>
        </footer>
      )}
    </li>
  );
}

function stageIcon(kind: GgufQuantWorkedStage["kind"]) {
  switch (kind) {
    case "source":
      return <Calculator aria-hidden="true" />;
    case "metadata":
      return <Binary aria-hidden="true" />;
    case "storage":
      return <Boxes aria-hidden="true" />;
    case "reconstruction":
      return <Cpu aria-hidden="true" />;
  }
}

function formatDecimal(value: number): string {
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}
