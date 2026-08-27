(function () {
    "use strict";

    const korean = (document.documentElement.lang || "en").slice(0, 2) === "ko";

    function initializeKappaDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-kappa]"));
        const output = demo.querySelector("[data-kappa-output]");
        const note = demo.querySelector("[data-kappa-note]");
        const path = demo.querySelector("[data-step-path]");
        const contours = ["outer", "middle", "inner"].map(function (name) {
            return demo.querySelector('[data-contour="' + name + '"]');
        });
        const copy = korean ? {
            1: "방향들이 균형을 이룹니다. 하나의 step size로 각 방향에서 유용하게 움직일 수 있습니다.",
            10: "골짜기가 기울기 시작합니다. 좌표별 scale 조정만으로는 일부만 보정됩니다.",
            100: "좁은 방향에 안전한 step은 긴 방향을 따라서는 고통스러울 만큼 느립니다.",
            1000: "정보는 그대로지만, 유한한 최적화는 길고 강하게 결합된 골짜기를 만납니다."
        } : {
            1: "Directions are balanced. One step size can make useful progress in each.",
            10: "The valley becomes oblique. Coordinate-wise scaling only partly follows it.",
            100: "A step that is safe across the narrow direction is painfully slow along the valley.",
            1000: "The information is unchanged, but finite optimization now faces a long, coupled valley."
        };
        const paths = {
            1: "M118 266 C210 230 288 194 382 166",
            10: "M118 266 L188 245 L232 257 L283 214 L326 222 L382 166",
            100: "M118 266 L171 236 L205 260 L246 218 L280 238 L316 194 L345 208 L382 166",
            1000: "M118 266 L158 232 L190 265 L221 224 L252 246 L279 207 L307 228 L331 192 L353 207 L382 166"
        };

        function select(button) {
            const kappa = Number(button.dataset.kappa);
            const stress = Math.log10(kappa) / 3;
            const sizes = [
                [112 + 188 * stress, 88 - 63 * stress],
                [78 + 132 * stress, 61 - 44 * stress],
                [43 + 76 * stress, 34 - 25 * stress]
            ];

            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            contours.forEach(function (ellipse, index) {
                ellipse.setAttribute("rx", sizes[index][0].toFixed(1));
                ellipse.setAttribute("ry", sizes[index][1].toFixed(1));
            });
            path.setAttribute("d", paths[kappa]);
            output.textContent = kappa.toLocaleString(korean ? "ko-KR" : "en-US") + "×";
            note.textContent = copy[kappa];
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    function initializeTrajectoryDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-trajectory]"));
        const changedPath = demo.querySelector('[data-trajectory-path="changed"]');
        const changedStart = demo.querySelector("[data-trajectory-start]");
        const changedLabel = demo.querySelector("[data-trajectory-label]");
        const harm = demo.querySelector("[data-trajectory-harm]");
        const drift = demo.querySelector("[data-trajectory-drift]");
        const driftLabel = demo.querySelector("[data-trajectory-drift-label]");
        const note = demo.querySelector("[data-trajectory-note]");
        const functionLevel = demo.querySelector("[data-level-function]");
        const pathLevel = demo.querySelector("[data-level-path]");
        const states = korean ? {
            ordinary: {
                path: "M100 245 C245 232 430 203 650 170", start: 245, labelY: 191,
                label: "변환 좌표", harm: "54.00%", driftLabel: "step-0 함수", drift: "다름",
                note: "같은 정보라도 일반 무작위 초기화가 같은 함수를 뽑는 것은 아닙니다.",
                functionLevel: "다른 시작", pathLevel: "다른 경로", matchedFunction: false, matchedPath: false
            },
            matched: {
                path: "M100 210 C235 198 405 154 650 125", start: 210, labelY: 146,
                label: "변환 좌표", harm: "2.89%", driftLabel: "step-1 drift", drift: "0.071",
                note: "두 함수는 같게 시작하지만 AdamW update 한 번 뒤 모든 통제 pair가 벌어집니다.",
                functionLevel: "같은 시작", pathLevel: "다른 경로", matchedFunction: true, matchedPath: false
            },
            natural: {
                path: "M100 210 C235 184 405 107 650 64", start: 210, labelY: 87,
                label: "변환 좌표", harm: "0.00002%", driftLabel: "step-200 drift", drift: "0.000076",
                note: "Full-rank 통제 pair에서는 matched input-natural update가 두 경로를 거의 겹치게 합니다.",
                functionLevel: "같은 시작", pathLevel: "같은 경로*", matchedFunction: true, matchedPath: true
            }
        } : {
            ordinary: {
                path: "M100 245 C245 232 430 203 650 170", start: 245, labelY: 191,
                label: "recoded", harm: "54.00%", driftLabel: "step-0 functions", drift: "different",
                note: "Equal information does not make ordinary random initialization sample the same function.",
                functionLevel: "different start", pathLevel: "different path", matchedFunction: false, matchedPath: false
            },
            matched: {
                path: "M100 210 C235 198 405 154 650 125", start: 210, labelY: 146,
                label: "recoded", harm: "2.89%", driftLabel: "step-1 drift", drift: "0.071",
                note: "The functions start equal, yet one AdamW update makes every controlled pair move apart.",
                functionLevel: "same start", pathLevel: "different path", matchedFunction: true, matchedPath: false
            },
            natural: {
                path: "M100 210 C235 184 405 107 650 64", start: 210, labelY: 87,
                label: "recoded", harm: "0.00002%", driftLabel: "step-200 drift", drift: "0.000076",
                note: "On full-rank controlled pairs, the matched input-natural update nearly overlaps both paths.",
                functionLevel: "same start", pathLevel: "same path*", matchedFunction: true, matchedPath: true
            }
        };

        function select(button) {
            const state = states[button.dataset.trajectory];
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            changedPath.setAttribute("d", state.path);
            changedStart.setAttribute("cy", state.start);
            changedLabel.setAttribute("y", state.labelY);
            changedLabel.textContent = state.label;
            harm.textContent = state.harm;
            driftLabel.textContent = state.driftLabel;
            drift.textContent = state.drift;
            note.textContent = state.note;
            functionLevel.textContent = state.functionLevel;
            pathLevel.textContent = state.pathLevel;
            functionLevel.classList.toggle("kept", state.matchedFunction);
            pathLevel.classList.toggle("kept", state.matchedPath);
            demo.dataset.activeTrajectory = button.dataset.trajectory;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    initializeKappaDemo(document.querySelector("[data-kappa-demo]"));
    initializeTrajectoryDemo(document.querySelector("[data-trajectory-demo]"));
}());
