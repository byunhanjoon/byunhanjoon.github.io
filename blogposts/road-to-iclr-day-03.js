(function () {
    "use strict";

    const korean = (document.documentElement.lang || "en").slice(0, 2) === "ko";

    function initializeOrbitDemo(demo) {
        if (!demo) return;

        const table = demo.querySelector("[data-orbit-table]");
        const headers = Array.from(table.querySelectorAll(".d3-table-grid > b"));
        const cells = Array.from(table.querySelectorAll(".d3-table-grid > i"));
        const name = table.querySelector("[data-orbit-name]");
        const note = table.querySelector("[data-orbit-note]");
        const before = table.querySelector("[data-orbit-before]");
        const after = table.querySelector("[data-orbit-after]");
        const controls = Array.from(demo.querySelectorAll("[data-orbit-view]"));
        const states = korean ? {
            original: {
                name: "원본",
                headers: ["나이", "플랜", "소득"],
                cells: ["37", "실버", "$52k", "54", "골드", "$81k"],
                note: "표기를 눌러 보세요. 행과 target은 그대로입니다.",
                before: "나이 · 플랜 · 소득", after: "변화 없음",
                changedHeaders: [], changedCells: []
            },
            columns: {
                name: "열 순서",
                headers: ["소득", "나이", "플랜"],
                cells: ["$52k", "37", "실버", "$81k", "54", "골드"],
                note: "순서만 바뀌었습니다. 값은 각 열과 함께 이동합니다.",
                before: "나이 · 플랜 · 소득", after: "소득 · 나이 · 플랜",
                changedHeaders: [0, 1, 2], changedCells: [0, 1, 2, 3, 4, 5]
            },
            categories: {
                name: "범주 이름",
                headers: ["나이", "플랜 ID", "소득"],
                cells: ["37", "0", "$52k", "54", "1", "$81k"],
                note: "범주 이름만 바뀌었습니다. 실버↔0, 골드↔1입니다.",
                before: "실버 · 골드", after: "0 · 1",
                changedHeaders: [1], changedCells: [1, 4]
            },
            units: {
                name: "센트 단위",
                headers: ["나이", "플랜", "소득 (¢)"],
                cells: ["37", "실버", "5.2M¢", "54", "골드", "8.1M¢"],
                note: "단위만 바뀌었습니다. 달러를 센트로 표시합니다.",
                before: "$52k · $81k", after: "5.2M¢ · 8.1M¢",
                changedHeaders: [2], changedCells: [2, 5]
            },
            basis: {
                name: "혼합 기저",
                headers: ["z₁", "z₂", "z₃"],
                cells: ["89", "37", "52", "135", "55", "82"],
                note: "좌표만 바뀌었습니다. 가역적 혼합이라 정보는 그대로입니다.",
                before: "나이 · 플랜 · 소득", after: "z₁ · z₂ · z₃",
                changedHeaders: [0, 1, 2], changedCells: [0, 1, 2, 3, 4, 5]
            }
        } : {
            original: {
                name: "original",
                headers: ["age", "plan", "income"],
                cells: ["37", "silver", "$52k", "54", "gold", "$81k"],
                note: "Choose a spelling; the rows and target stay fixed.",
                before: "age · plan · income", after: "unchanged",
                changedHeaders: [], changedCells: []
            },
            columns: {
                name: "column order",
                headers: ["income", "age", "plan"],
                cells: ["$52k", "37", "silver", "$81k", "54", "gold"],
                note: "Only the order changed; values moved with their columns.",
                before: "age · plan · income", after: "income · age · plan",
                changedHeaders: [0, 1, 2], changedCells: [0, 1, 2, 3, 4, 5]
            },
            categories: {
                name: "category names",
                headers: ["age", "plan ID", "income"],
                cells: ["37", "0", "$52k", "54", "1", "$81k"],
                note: "Only the category names changed: silver↔0 and gold↔1.",
                before: "silver · gold", after: "0 · 1",
                changedHeaders: [1], changedCells: [1, 4]
            },
            units: {
                name: "cents",
                headers: ["age", "plan", "income (¢)"],
                cells: ["37", "silver", "5.2M¢", "54", "gold", "8.1M¢"],
                note: "Only the units changed: dollars are now shown as cents.",
                before: "$52k · $81k", after: "5.2M¢ · 8.1M¢",
                changedHeaders: [2], changedCells: [2, 5]
            },
            basis: {
                name: "mixed basis",
                headers: ["z₁", "z₂", "z₃"],
                cells: ["89", "37", "52", "135", "55", "82"],
                note: "Only the coordinates changed; an invertible mix keeps all information.",
                before: "age · plan · income", after: "z₁ · z₂ · z₃",
                changedHeaders: [0, 1, 2], changedCells: [0, 1, 2, 3, 4, 5]
            }
        };

        function select(control, animate) {
            const state = states[control.dataset.orbitView];
            controls.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === control ? "true" : "false");
            });
            headers.forEach(function (cell, index) { cell.textContent = state.headers[index]; });
            cells.forEach(function (cell, index) { cell.textContent = state.cells[index]; });
            headers.concat(cells).forEach(function (cell) { cell.classList.remove("is-changed"); });
            state.changedHeaders.forEach(function (index) { headers[index].classList.add("is-changed"); });
            state.changedCells.forEach(function (index) { cells[index].classList.add("is-changed"); });
            name.textContent = state.name;
            note.textContent = state.note;
            before.textContent = state.before;
            after.textContent = state.after;
            demo.dataset.activeOrbit = control.dataset.orbitView;
            if (animate) {
                table.classList.remove("is-changing");
                void table.offsetWidth;
                table.classList.add("is-changing");
            }
        }

        controls.forEach(function (control) {
            control.addEventListener("click", function () { select(control, true); });
        });
        select(controls.find(function (control) {
            return control.getAttribute("aria-pressed") === "true";
        }) || controls[0], false);
    }

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

    initializeOrbitDemo(document.querySelector("[data-orbit-demo]"));
    initializeKappaDemo(document.querySelector("[data-kappa-demo]"));
    initializeTrajectoryDemo(document.querySelector("[data-trajectory-demo]"));
}());
