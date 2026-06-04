import { Composition } from "remotion";
import { TrajectoryReplay } from "./TrajectoryReplay";
import cases from "./data/cases.json";

const FPS = 30;
const DURATION_SECONDS = 40;
const DURATION = FPS * DURATION_SECONDS;
const WIDTH = 1280;
const HEIGHT = 720;

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {cases.cases.map((c) => (
        <Composition
          key={`${c.benchmark}-${c.id}`}
          id={`Trajectory-${c.benchmark}-${c.id}`}
          component={TrajectoryReplay as any}
          durationInFrames={DURATION}
          fps={FPS}
          width={WIDTH}
          height={HEIGHT}
          defaultProps={{ caseData: c }}
        />
      ))}
    </>
  );
};
