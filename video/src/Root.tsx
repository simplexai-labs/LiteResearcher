import { Composition } from "remotion";
import { TrajectoryReplay } from "./TrajectoryReplay";
import cases from "./data/cases.json";

const FPS = 30;
const WIDTH = 1280;
const HEIGHT = 720;
const HEAD_FRAMES = 60;
const TAIL_FRAMES = 360;

// Mirrors computeTurnFrames in TrajectoryReplay
const turnFrames = (t: any): number => {
  const thinkChars = Math.min((t.think || "").length, 8 * 110);
  const toolChars = t.tool
    ? t.tool.name === "search"
      ? (t.tool.queries || []).join("").length
      : t.tool.name === "visit"
        ? (t.tool.urls || []).join("").length + (t.tool.goal?.length || 0)
        : 0
    : 0;
  const thinkF = Math.ceil(thinkChars / 80) + 4;
  const toolF = 4 + Math.ceil(toolChars / 36) + 3;
  const resultF = t.tool?.result ? 26 : 0;
  const pauseF = 4;
  return thinkF + toolF + resultF + pauseF;
};

const durationFor = (c: any) =>
  HEAD_FRAMES + c.turns.reduce((acc: number, t: any) => acc + turnFrames(t), 0) + TAIL_FRAMES;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {cases.cases.map((c) => (
        <Composition
          key={`${c.benchmark}-${c.id}`}
          id={`Trajectory-${c.benchmark}-${c.id}`}
          component={TrajectoryReplay as any}
          durationInFrames={durationFor(c)}
          fps={FPS}
          width={WIDTH}
          height={HEIGHT}
          defaultProps={{ caseData: c }}
        />
      ))}
    </>
  );
};
