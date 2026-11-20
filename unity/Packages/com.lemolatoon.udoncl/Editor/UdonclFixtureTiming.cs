using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading;
using UnityEngine;

namespace Lemolatoon.Udoncl.Editor
{
    // Wall-clock, frame and readback-latency measurement for the Editor
    // fixtures that drive a generated U# binding from a frame-polling loop.
    //
    // A batchmode Editor renders with no vsync and no frame-rate limit, so a
    // frame count there is not a proxy for time: near-empty frames spin while
    // an asynchronous GPU readback waits for the GPU backlog. Every figure is
    // therefore reported in frames and in milliseconds of
    // Time.realtimeSinceStartup, and the optional fixed-rate mode holds the
    // frame period of a VRChat client so that its frame counts are comparable.
    //
    // The fixture observes the controller once per frame, so every boundary
    // (phase change, readback completion) has one-frame resolution.
    //
    // Log lines (one Debug.Log each, fields separated by single spaces):
    //   udoncl <label> timing[ <context>] mode=<unbounded|fixed> target_fps=..
    //   udoncl <label> timing[ <context>] phase=<name> frames=.. ms=..
    //       steps=.. [ms_per_step=.. frames_per_step=..]
    //   udoncl <label> readback[ <context>] index=.. name=.. frames=.. ms=..
    //   udoncl <label> timing[ <context>] total frames=.. unity_frames=..
    //       ms=.. fps=.. mode=.. target_fps=.. paced_frames=.. readbacks=..
    //       [readback_frames_min/max/mean=.. readback_ms_min/max/mean=..]
    internal sealed class UdonclFixtureTiming
    {
        // Unset, empty or 0: leave the Editor frame rate alone ("unbounded").
        // A positive integer N: QualitySettings.vSyncCount = 0 and
        // Application.targetFrameRate = N for the run, restored by End().
        public const string FrameRateVariable = "UDONCL_FIXTURE_TARGET_FRAME_RATE";

        private readonly string label;
        private readonly string context;
        private readonly int targetFrameRate;
        private readonly Dictionary<string, int> phaseFrames = new Dictionary<string, int>();
        private readonly Dictionary<string, double> phaseMilliseconds =
            new Dictionary<string, double>();

        private bool running;
        private bool ended;
        private int savedVSyncCount;
        private int savedTargetFrameRate;
        private double startTime;
        private double endTime;
        private double lastFrameTime;
        private int startUnityFrame;
        private int endUnityFrame;
        private int frames;
        private int pacedFrames;

        private string phaseName;
        private int phaseSteps;
        private int phaseStartFrame;
        private double phaseStartTime;

        private bool readbackOpen;
        private string readbackName;
        private int readbackStartFrame;
        private double readbackStartTime;
        private int readbackCount;
        private int readbackMeasured;
        private int readbackFramesMin;
        private int readbackFramesMax;
        private long readbackFramesSum;
        private double readbackMsMin;
        private double readbackMsMax;
        private double readbackMsSum;

        public UdonclFixtureTiming(string label)
            : this(label, null) { }

        // `context` is an optional "key=value" field (for example
        // "step=step-000") repeated on every line of this measurement.
        public UdonclFixtureTiming(string label, string context)
        {
            this.label = label;
            this.context = string.IsNullOrEmpty(context) ? "" : " " + context;
            targetFrameRate = RequestedFrameRate();
        }

        public static int RequestedFrameRate()
        {
            string text = Environment.GetEnvironmentVariable(FrameRateVariable);
            int value = 0;
            if (
                string.IsNullOrWhiteSpace(text)
                || !int.TryParse(
                    text.Trim(),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out value
                )
                || value <= 0
            )
                return 0;
            return value;
        }

        public string Mode
        {
            get { return targetFrameRate > 0 ? "fixed" : "unbounded"; }
        }

        public int TargetFrameRate
        {
            get { return targetFrameRate; }
        }

        public int Frames
        {
            get { return frames; }
        }

        public double ElapsedMilliseconds
        {
            get
            {
                if (!running && !ended)
                    return 0.0;
                return ((ended ? endTime : Now()) - startTime) * 1000.0;
            }
        }

        public string ElapsedMillisecondsText
        {
            get { return Ms(ElapsedMilliseconds); }
        }

        // Fields appended to a fixture's existing result line.
        public string Summary
        {
            get
            {
                double ms = ElapsedMilliseconds;
                return " ms="
                    + Ms(ms)
                    + " fps="
                    + Ms(ms > 0.0 ? frames * 1000.0 / ms : 0.0)
                    + " mode="
                    + Mode
                    + " target_fps="
                    + targetFrameRate.ToString(CultureInfo.InvariantCulture);
            }
        }

        // Starts the clocks, applies the frame-rate mode and opens the first
        // phase. Call immediately before the controller is started.
        public void Begin(string firstPhase, int steps)
        {
            if (running || ended)
                return;
            savedVSyncCount = QualitySettings.vSyncCount;
            savedTargetFrameRate = Application.targetFrameRate;
            if (targetFrameRate > 0)
            {
                QualitySettings.vSyncCount = 0;
                Application.targetFrameRate = targetFrameRate;
            }
            running = true;
            startTime = Now();
            lastFrameTime = startTime;
            startUnityFrame = Time.frameCount;
            frames = 0;
            pacedFrames = 0;
            phaseName = firstPhase;
            phaseSteps = steps;
            phaseStartFrame = 0;
            phaseStartTime = startTime;
            UnityEngine.Debug.Log(
                "udoncl "
                    + label
                    + " timing"
                    + context
                    + " mode="
                    + Mode
                    + " target_fps="
                    + targetFrameRate.ToString(CultureInfo.InvariantCulture)
                    + " vsync="
                    + QualitySettings.vSyncCount.ToString(CultureInfo.InvariantCulture)
                    + " previous_target_fps="
                    + savedTargetFrameRate.ToString(CultureInfo.InvariantCulture)
                    + " previous_vsync="
                    + savedVSyncCount.ToString(CultureInfo.InvariantCulture)
                    + " batchmode="
                    + (Application.isBatchMode ? "true" : "false")
                    + " graphics="
                    + SystemInfo.graphicsDeviceType
            );
        }

        // Call once after every `yield return null` of the polling loop,
        // before the controller is observed. In fixed mode a frame that
        // finished early is held until its period has elapsed: a batchmode
        // Editor does not necessarily honor Application.targetFrameRate, and
        // paced_frames reports how many frames this fallback had to hold.
        public void Frame()
        {
            if (!running)
                return;
            ++frames;
            if (targetFrameRate > 0)
            {
                double period = 1.0 / targetFrameRate;
                bool paced = false;
                double remaining = period - (Now() - lastFrameTime);
                while (remaining > 0.0)
                {
                    paced = true;
                    if (remaining > 0.002)
                        Thread.Sleep(1);
                    else
                        Thread.SpinWait(64);
                    remaining = period - (Now() - lastFrameTime);
                }
                if (paced)
                    ++pacedFrames;
            }
            lastFrameTime = Now();
        }

        // Closes the open phase and opens `name`. A repeated name is ignored,
        // so the fixture may report the observed phase on every change.
        // `steps` is the number of model steps the phase executes (0 when the
        // phase is not a model run); it only adds the per-step fields.
        public void Phase(string name, int steps)
        {
            if (!running || name == phaseName)
                return;
            ClosePhase();
            phaseName = name;
            phaseSteps = steps;
            phaseStartFrame = frames;
            phaseStartTime = Now();
        }

        // Frame-polled readback observation. `completed` is the number of
        // readbacks whose callback has run so far and `pending` says whether
        // one is in flight now; `nextName` names the readback in flight. The
        // latency of each readback is request -> callback as seen by the
        // per-frame poll.
        public void Readbacks(int completed, bool pending, string nextName)
        {
            if (!running)
                return;
            while (readbackOpen && readbackCount < completed)
                CloseReadback();
            if (readbackCount < completed)
                readbackCount = completed;
            if (pending && !readbackOpen)
            {
                readbackOpen = true;
                readbackName = nextName;
                readbackStartFrame = frames;
                readbackStartTime = Now();
            }
        }

        public int PhaseFrames(string name)
        {
            int value;
            return phaseFrames.TryGetValue(name, out value) ? value : 0;
        }

        public double PhaseMilliseconds(string name)
        {
            double value;
            return phaseMilliseconds.TryGetValue(name, out value) ? value : 0.0;
        }

        public string PhaseMillisecondsText(string name)
        {
            return Ms(PhaseMilliseconds(name));
        }

        // Closes the open phase, restores the frame-rate settings and logs
        // the total. Idempotent, so it belongs in a `finally` block.
        public void End()
        {
            if (!running)
                return;
            ClosePhase();
            endTime = Now();
            endUnityFrame = Time.frameCount;
            running = false;
            ended = true;
            QualitySettings.vSyncCount = savedVSyncCount;
            Application.targetFrameRate = savedTargetFrameRate;
            StringBuilder line = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" timing")
                .Append(context)
                .Append(" total frames=")
                .Append(frames.ToString(CultureInfo.InvariantCulture))
                .Append(" unity_frames=")
                .Append((endUnityFrame - startUnityFrame).ToString(CultureInfo.InvariantCulture))
                .Append(Summary)
                .Append(" paced_frames=")
                .Append(pacedFrames.ToString(CultureInfo.InvariantCulture))
                .Append(" readbacks=")
                .Append(readbackCount.ToString(CultureInfo.InvariantCulture));
            if (readbackMeasured > 0)
                line.Append(" readback_frames_min=")
                    .Append(readbackFramesMin.ToString(CultureInfo.InvariantCulture))
                    .Append(" readback_frames_max=")
                    .Append(readbackFramesMax.ToString(CultureInfo.InvariantCulture))
                    .Append(" readback_frames_mean=")
                    .Append(Ms((double)readbackFramesSum / readbackMeasured))
                    .Append(" readback_ms_min=")
                    .Append(Ms(readbackMsMin))
                    .Append(" readback_ms_max=")
                    .Append(Ms(readbackMsMax))
                    .Append(" readback_ms_mean=")
                    .Append(Ms(readbackMsSum / readbackMeasured));
            UnityEngine.Debug.Log(line.ToString());
        }

        private void ClosePhase()
        {
            if (phaseName == null)
                return;
            int elapsedFrames = frames - phaseStartFrame;
            double ms = (Now() - phaseStartTime) * 1000.0;
            int totalFrames;
            double totalMs;
            phaseFrames.TryGetValue(phaseName, out totalFrames);
            phaseMilliseconds.TryGetValue(phaseName, out totalMs);
            phaseFrames[phaseName] = totalFrames + elapsedFrames;
            phaseMilliseconds[phaseName] = totalMs + ms;
            StringBuilder line = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" timing")
                .Append(context)
                .Append(" phase=")
                .Append(phaseName)
                .Append(" frames=")
                .Append(elapsedFrames.ToString(CultureInfo.InvariantCulture))
                .Append(" ms=")
                .Append(Ms(ms))
                .Append(" steps=")
                .Append(phaseSteps.ToString(CultureInfo.InvariantCulture));
            if (phaseSteps > 0)
                line.Append(" ms_per_step=")
                    .Append(Ms(ms / phaseSteps))
                    .Append(" frames_per_step=")
                    .Append(Ms((double)elapsedFrames / phaseSteps));
            UnityEngine.Debug.Log(line.ToString());
            phaseName = null;
        }

        private void CloseReadback()
        {
            int elapsedFrames = frames - readbackStartFrame;
            double ms = (Now() - readbackStartTime) * 1000.0;
            if (readbackMeasured == 0 || elapsedFrames < readbackFramesMin)
                readbackFramesMin = elapsedFrames;
            if (readbackMeasured == 0 || elapsedFrames > readbackFramesMax)
                readbackFramesMax = elapsedFrames;
            if (readbackMeasured == 0 || ms < readbackMsMin)
                readbackMsMin = ms;
            if (readbackMeasured == 0 || ms > readbackMsMax)
                readbackMsMax = ms;
            ++readbackMeasured;
            readbackFramesSum += elapsedFrames;
            readbackMsSum += ms;
            UnityEngine.Debug.Log(
                "udoncl "
                    + label
                    + " readback"
                    + context
                    + " index="
                    + readbackCount.ToString(CultureInfo.InvariantCulture)
                    + " name="
                    + (string.IsNullOrEmpty(readbackName) ? "unnamed" : readbackName)
                    + " frames="
                    + elapsedFrames.ToString(CultureInfo.InvariantCulture)
                    + " ms="
                    + Ms(ms)
            );
            ++readbackCount;
            readbackOpen = false;
        }

        private static double Now()
        {
            return Time.realtimeSinceStartupAsDouble;
        }

        private static string Ms(double value)
        {
            return value.ToString("F3", CultureInfo.InvariantCulture);
        }
    }
}
