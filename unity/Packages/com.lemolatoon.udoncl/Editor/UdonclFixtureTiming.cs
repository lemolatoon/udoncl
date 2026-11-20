using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Threading;
using UnityEngine;
using VRC.Udon;

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
    //
    // Fixed mode only (the same UDONCL_FIXTURE_TARGET_FRAME_RATE switch),
    // written by End() after the total line:
    //   udoncl <label> frame_timing[ <context>] available=.. samples=..
    //       gpu_ms_mean/max=.. main_ms_mean/max=.. render_ms_mean/max=..
    //   and, for a binding passed to TraceBinding (the opt-in pump trace of
    //   the generated binding, see DriveAsyncCommands):
    //   udoncl <label> pump[ <context>] trace=<on|unavailable>
    //       continuations=.. in_window=.. first_of_frame=.. unity_frames=..
    //       continuations_per_frame_mean=.. continuations_per_frame_max=..
    //       event_us_total/mean/max=.. frame_event_us_mean/max=..
    //       windows=.. stop_limit=.. stop_budget=.. stop_estimate=.. stop_idle=..
    //       window_us_mean/max=..
    //   udoncl <label> pump_continuations[ <context>] <frame>:<driven>:<us>,...
    //   udoncl <label> pump_windows[ <context>] <frame>:<count>:<us>:<reason>,...
    //   (driven 0 for the first Host slice charged to its frame's window;
    //   reason 1 continuation limit, 2 time budget, 3 nothing ready,
    //   4 adaptive estimate of the next slice exceeds the budget)
    //
    // Any mode, written by TraceBinding for a binding with asynchronous
    // exports (the run-time frame budget of the generated binding, after the
    // optional UDONCL_FIXTURE_FRAME_* overrides below):
    //   udoncl <label> pump_budget[ <context>] budget_us=.. continuations=..
    //       adaptive=<true|false> override=<true|false>
    internal sealed class UdonclFixtureTiming
    {
        // Unset, empty or 0: leave the Editor frame rate alone ("unbounded").
        // A positive integer N: QualitySettings.vSyncCount = 0 and
        // Application.targetFrameRate = N for the run, restored by End().
        public const string FrameRateVariable = "UDONCL_FIXTURE_TARGET_FRAME_RATE";

        // Optional run-time frame budget of the traced binding (what U#
        // sets through TrySetAsyncFrameBudget and SetAsyncFrameAdaptive).
        // Unset or empty keeps the binding's manifest defaults. Budget and
        // continuations take an integer from 0 to 1000000; adaptive takes
        // 0 or 1. Any other value fails the fixture.
        public const string FrameBudgetVariable = "UDONCL_FIXTURE_FRAME_BUDGET_US";
        public const string FrameContinuationsVariable = "UDONCL_FIXTURE_FRAME_CONTINUATIONS";
        public const string FrameAdaptiveVariable = "UDONCL_FIXTURE_FRAME_ADAPTIVE";
        private const int FrameSettingMaximum = 1000000;

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

        // Fixed mode only: FrameTimingManager samples (milliseconds) and the
        // generated binding whose opt-in pump trace End() reports.
        private const int TraceContinuationCapacity = 16384;
        private const int TraceWindowCapacity = 16384;
        private readonly FrameTiming[] frameTimingSample = new FrameTiming[1];
        private readonly List<double> gpuMilliseconds = new List<double>();
        private readonly List<double> mainMilliseconds = new List<double>();
        private readonly List<double> renderMilliseconds = new List<double>();
        private UdonBehaviour tracedBinding;
        private bool bindingTraceOn;

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
                SampleFrameTiming();
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
            if (targetFrameRate > 0)
            {
                ReportFrameTiming();
                ReportBindingTrace();
            }
        }

        // Applies the optional UDONCL_FIXTURE_FRAME_* overrides to a
        // generated binding that has asynchronous exports and logs its
        // run-time frame budget; in fixed mode it also turns on the binding's
        // opt-in pump trace by assigning its trace arrays. Call before the
        // controller starts the first export. A binding without the trace
        // fields reports trace=unavailable.
        public void TraceBinding(UdonBehaviour binding)
        {
            if (binding == null)
                return;
            ApplyFrameBudget(binding);
            if (targetFrameRate <= 0)
                return;
            tracedBinding = binding;
            binding.SetProgramVariable(
                "asyncTraceContinuations",
                new int[TraceContinuationCapacity * 3]
            );
            binding.SetProgramVariable("asyncTraceContinuationWords", 0);
            binding.SetProgramVariable("asyncTraceWindows", new int[TraceWindowCapacity * 4]);
            binding.SetProgramVariable("asyncTraceWindowWords", 0);
            bindingTraceOn =
                binding.GetProgramVariable("asyncTraceContinuations") is int[]
                && binding.GetProgramVariable("asyncTraceWindows") is int[];
        }

        // The overrides assign the private fields behind the binding's
        // TrySetAsyncFrameBudget/SetAsyncFrameAdaptive (an Editor fixture
        // cannot call a U# method with arguments) after the same range
        // check, then read the effective values back.
        private void ApplyFrameBudget(UdonBehaviour binding)
        {
            int? budget = FrameSetting(FrameBudgetVariable, FrameSettingMaximum);
            int? continuations = FrameSetting(FrameContinuationsVariable, FrameSettingMaximum);
            int? adaptive = FrameSetting(FrameAdaptiveVariable, 1);
            bool overridden = budget.HasValue || continuations.HasValue || adaptive.HasValue;
            bool available =
                binding.GetProgramVariable("asyncFrameBudgetMicroseconds") is int
                && binding.GetProgramVariable("asyncFrameContinuationLimit") is int
                && binding.GetProgramVariable("asyncFrameAdaptive") is bool;
            if (!available)
            {
                if (overridden)
                    throw new InvalidOperationException(
                        "the generated binding has no run-time frame budget to override"
                    );
                return;
            }
            if (budget.HasValue)
                binding.SetProgramVariable("asyncFrameBudgetMicroseconds", budget.Value);
            if (continuations.HasValue)
                binding.SetProgramVariable("asyncFrameContinuationLimit", continuations.Value);
            if (adaptive.HasValue)
            {
                binding.SetProgramVariable("asyncFrameAdaptive", adaptive.Value != 0);
                binding.SetProgramVariable("asyncSliceEstimate", 0.0);
            }
            UnityEngine.Debug.Log(
                "udoncl "
                    + label
                    + " pump_budget"
                    + context
                    + " budget_us="
                    + ((int)binding.GetProgramVariable("asyncFrameBudgetMicroseconds")).ToString(
                        CultureInfo.InvariantCulture
                    )
                    + " continuations="
                    + ((int)binding.GetProgramVariable("asyncFrameContinuationLimit")).ToString(
                        CultureInfo.InvariantCulture
                    )
                    + " adaptive="
                    + ((bool)binding.GetProgramVariable("asyncFrameAdaptive") ? "true" : "false")
                    + " override="
                    + (overridden ? "true" : "false")
            );
        }

        private static int? FrameSetting(string variable, int maximum)
        {
            string text = Environment.GetEnvironmentVariable(variable);
            if (string.IsNullOrWhiteSpace(text))
                return null;
            int value;
            if (
                !int.TryParse(
                    text.Trim(),
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out value
                )
                || value < 0
                || value > maximum
            )
                throw new InvalidOperationException(
                    variable
                        + " must be an integer from 0 to "
                        + maximum.ToString(CultureInfo.InvariantCulture)
                        + ": "
                        + text
                );
            return value;
        }

        private void SampleFrameTiming()
        {
            FrameTimingManager.CaptureFrameTimings();
            if (FrameTimingManager.GetLatestTimings(1, frameTimingSample) == 0)
                return;
            gpuMilliseconds.Add(frameTimingSample[0].gpuFrameTime);
            mainMilliseconds.Add(frameTimingSample[0].cpuMainThreadFrameTime);
            renderMilliseconds.Add(frameTimingSample[0].cpuRenderThreadFrameTime);
        }

        private void ReportFrameTiming()
        {
            StringBuilder line = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" frame_timing")
                .Append(context)
                .Append(" available=")
                .Append(gpuMilliseconds.Count > 0 ? "true" : "false")
                .Append(" feature_enabled=")
                .Append(FrameTimingManager.IsFeatureEnabled() ? "true" : "false")
                .Append(" samples=")
                .Append(gpuMilliseconds.Count.ToString(CultureInfo.InvariantCulture));
            AppendMeanMax(line, "gpu_ms", gpuMilliseconds);
            AppendMeanMax(line, "main_ms", mainMilliseconds);
            AppendMeanMax(line, "render_ms", renderMilliseconds);
            UnityEngine.Debug.Log(line.ToString());
        }

        private static void AppendMeanMax(StringBuilder line, string name, List<double> values)
        {
            double sum = 0.0;
            double max = 0.0;
            foreach (double value in values)
            {
                sum += value;
                if (value > max)
                    max = value;
            }
            line.Append(' ')
                .Append(name)
                .Append("_mean=")
                .Append(Ms(values.Count > 0 ? sum / values.Count : 0.0))
                .Append(' ')
                .Append(name)
                .Append("_max=")
                .Append(Ms(max));
        }

        private void ReportBindingTrace()
        {
            if (tracedBinding == null)
                return;
            StringBuilder summary = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" pump")
                .Append(context)
                .Append(" trace=")
                .Append(bindingTraceOn ? "on" : "unavailable");
            int[] continuations =
                tracedBinding.GetProgramVariable("asyncTraceContinuations") as int[];
            int[] windows = tracedBinding.GetProgramVariable("asyncTraceWindows") as int[];
            object rawContinuationWords = tracedBinding.GetProgramVariable(
                "asyncTraceContinuationWords"
            );
            object rawWindowWords = tracedBinding.GetProgramVariable("asyncTraceWindowWords");
            if (
                !bindingTraceOn
                || continuations == null
                || windows == null
                || !(rawContinuationWords is int)
                || !(rawWindowWords is int)
            )
            {
                UnityEngine.Debug.Log(summary.ToString());
                return;
            }
            int continuationWords = Math.Min((int)rawContinuationWords, continuations.Length);
            int windowWords = Math.Min((int)rawWindowWords, windows.Length);

            int count = continuationWords / 3;
            int inWindow = 0;
            long eventTotal = 0;
            int eventMax = 0;
            int unityFrames = 0;
            int perFrameMax = 0;
            long frameEventMax = 0;
            int lastFrame = int.MinValue;
            int frameCount = 0;
            long frameEvent = 0;
            StringBuilder raw = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" pump_continuations")
                .Append(context)
                .Append(' ');
            for (int index = 0; index < count; ++index)
            {
                int frame = continuations[index * 3];
                int driven = continuations[index * 3 + 1];
                int us = continuations[index * 3 + 2];
                if (driven != 0)
                    ++inWindow;
                eventTotal += us;
                if (us > eventMax)
                    eventMax = us;
                if (frame != lastFrame)
                {
                    lastFrame = frame;
                    ++unityFrames;
                    frameCount = 0;
                    frameEvent = 0;
                }
                ++frameCount;
                frameEvent += us;
                if (frameCount > perFrameMax)
                    perFrameMax = frameCount;
                if (frameEvent > frameEventMax)
                    frameEventMax = frameEvent;
                if (index != 0)
                    raw.Append(',');
                raw.Append(frame.ToString(CultureInfo.InvariantCulture))
                    .Append(':')
                    .Append(driven.ToString(CultureInfo.InvariantCulture))
                    .Append(':')
                    .Append(us.ToString(CultureInfo.InvariantCulture));
            }

            int windowCount = windowWords / 4;
            int stopLimit = 0;
            int stopBudget = 0;
            int stopEstimate = 0;
            int stopIdle = 0;
            long windowTotal = 0;
            int windowMax = 0;
            StringBuilder rawWindows = new StringBuilder("udoncl ")
                .Append(label)
                .Append(" pump_windows")
                .Append(context)
                .Append(' ');
            for (int index = 0; index < windowCount; ++index)
            {
                int frame = windows[index * 4];
                int charged = windows[index * 4 + 1];
                int us = windows[index * 4 + 2];
                int reason = windows[index * 4 + 3];
                if (reason == 1)
                    ++stopLimit;
                else if (reason == 2)
                    ++stopBudget;
                else if (reason == 4)
                    ++stopEstimate;
                else
                    ++stopIdle;
                windowTotal += us;
                if (us > windowMax)
                    windowMax = us;
                if (index != 0)
                    rawWindows.Append(',');
                rawWindows
                    .Append(frame.ToString(CultureInfo.InvariantCulture))
                    .Append(':')
                    .Append(charged.ToString(CultureInfo.InvariantCulture))
                    .Append(':')
                    .Append(us.ToString(CultureInfo.InvariantCulture))
                    .Append(':')
                    .Append(reason.ToString(CultureInfo.InvariantCulture));
            }

            summary
                .Append(" continuations=")
                .Append(count.ToString(CultureInfo.InvariantCulture))
                .Append(" in_window=")
                .Append(inWindow.ToString(CultureInfo.InvariantCulture))
                .Append(" first_of_frame=")
                .Append((count - inWindow).ToString(CultureInfo.InvariantCulture))
                .Append(" unity_frames=")
                .Append(unityFrames.ToString(CultureInfo.InvariantCulture))
                .Append(" continuations_per_frame_mean=")
                .Append(Ms(unityFrames > 0 ? (double)count / unityFrames : 0.0))
                .Append(" continuations_per_frame_max=")
                .Append(perFrameMax.ToString(CultureInfo.InvariantCulture))
                .Append(" event_us_total=")
                .Append(eventTotal.ToString(CultureInfo.InvariantCulture))
                .Append(" event_us_mean=")
                .Append(Ms(count > 0 ? (double)eventTotal / count : 0.0))
                .Append(" event_us_max=")
                .Append(eventMax.ToString(CultureInfo.InvariantCulture))
                .Append(" frame_event_us_mean=")
                .Append(Ms(unityFrames > 0 ? (double)eventTotal / unityFrames : 0.0))
                .Append(" frame_event_us_max=")
                .Append(frameEventMax.ToString(CultureInfo.InvariantCulture))
                .Append(" windows=")
                .Append(windowCount.ToString(CultureInfo.InvariantCulture))
                .Append(" stop_limit=")
                .Append(stopLimit.ToString(CultureInfo.InvariantCulture))
                .Append(" stop_budget=")
                .Append(stopBudget.ToString(CultureInfo.InvariantCulture))
                .Append(" stop_estimate=")
                .Append(stopEstimate.ToString(CultureInfo.InvariantCulture))
                .Append(" stop_idle=")
                .Append(stopIdle.ToString(CultureInfo.InvariantCulture))
                .Append(" window_us_mean=")
                .Append(Ms(windowCount > 0 ? (double)windowTotal / windowCount : 0.0))
                .Append(" window_us_max=")
                .Append(windowMax.ToString(CultureInfo.InvariantCulture));
            UnityEngine.Debug.Log(summary.ToString());
            UnityEngine.Debug.Log(raw.ToString());
            UnityEngine.Debug.Log(rawWindows.ToString());
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
