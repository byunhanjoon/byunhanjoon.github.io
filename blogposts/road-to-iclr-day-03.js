(function () {
    "use strict";

    const isKorean = document.documentElement.lang === "ko";

    function initializeSchemaDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-schema]"));
        const table = demo.querySelector("[data-schema-table]");
        const label = demo.querySelector("[data-schema-label]");
        const prediction = demo.querySelector("[data-schema-prediction]");
        const meter = demo.querySelector("[data-schema-meter]");
        const marker = demo.querySelector("[data-schema-marker]");
        const note = demo.querySelector("[data-schema-note]");
        const states = isKorean ? {
            reference: {
                label: "기준 표기",
                heads: ["나이", "플랜", "소득"],
                rows: [["37", "실버", "$52k"], ["54", "골드", "$81k"]],
                prediction: 62,
                note: "재표현하지 않았습니다. 우연히 출발점으로 삼은 임의의 기준일 뿐입니다."
            },
            columns: {
                label: "열 이동",
                heads: ["소득", "나이", "플랜"],
                rows: [["$52k", "37", "실버"], ["$81k", "54", "골드"]],
                prediction: 68,
                note: "열의 위치만 바뀌었습니다. 각 셀은 여전히 같은 사람을 설명합니다."
            },
            categories: {
                label: "범주명 변경",
                heads: ["나이", "플랜 ID", "소득"],
                rows: [["37", "1", "$52k"], ["54", "0", "$81k"]],
                prediction: 55,
                note: "골드 → 0, 실버 → 1. 이름만 바뀌었고 범주 소속은 그대로입니다."
            },
            units: {
                label: "단위 변경",
                heads: ["나이", "플랜", "소득 (¢)"],
                rows: [["37", "실버", "5,200,000"], ["54", "골드", "8,100,000"]],
                prediction: 61,
                note: "되돌릴 수 있는 단위 변환으로 달러를 센트로 바꾸었습니다."
            }
        } : {
            reference: {
                label: "REFERENCE SPELLING",
                heads: ["age", "plan", "income"],
                rows: [["37", "silver", "$52k"], ["54", "gold", "$81k"]],
                prediction: 62,
                note: "No rewrite. This is the arbitrary reference we happened to start from."
            },
            columns: {
                label: "COLUMNS MOVED",
                heads: ["income", "age", "plan"],
                rows: [["$52k", "37", "silver"], ["$81k", "54", "gold"]],
                prediction: 68,
                note: "Only the column positions changed. Each cell still describes the same person."
            },
            categories: {
                label: "CATEGORIES RENAMED",
                heads: ["age", "plan ID", "income"],
                rows: [["37", "1", "$52k"], ["54", "0", "$81k"]],
                prediction: 55,
                note: "gold → 0 and silver → 1. The names changed; category membership did not."
            },
            units: {
                label: "UNITS CHANGED",
                heads: ["age", "plan", "income (¢)"],
                rows: [["37", "silver", "5,200,000"], ["54", "gold", "8,100,000"]],
                prediction: 61,
                note: "Dollars became cents through a reversible unit conversion."
            }
        };

        function renderTable(state) {
            table.innerHTML = "";
            const head = document.createElement("thead");
            const headRow = document.createElement("tr");
            state.heads.forEach(function (value) {
                const cell = document.createElement("th");
                cell.textContent = value;
                headRow.appendChild(cell);
            });
            head.appendChild(headRow);
            table.appendChild(head);

            const body = document.createElement("tbody");
            state.rows.forEach(function (row) {
                const tr = document.createElement("tr");
                row.forEach(function (value) {
                    const cell = document.createElement("td");
                    cell.textContent = value;
                    tr.appendChild(cell);
                });
                body.appendChild(tr);
            });
            table.appendChild(body);
        }

        function select(button) {
            const state = states[button.dataset.schema];
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            renderTable(state);
            label.textContent = state.label;
            prediction.textContent = state.prediction + "%";
            meter.style.width = state.prediction + "%";
            marker.style.left = state.prediction + "%";
            note.textContent = state.note;
            demo.dataset.schema = button.dataset.schema;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    function initializeTaxDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-spread]"));
        const dots = Array.from(demo.querySelectorAll("[data-tax-dot]"));
        const centroid = demo.querySelector("[data-tax-centroid]");
        const memberLoss = demo.querySelector("[data-member-loss]");
        const centroidLoss = demo.querySelector("[data-centroid-loss]");
        const schemaTax = demo.querySelector("[data-schema-tax]");
        const states = {
            steady: { values: [58, 59, 58, 59] },
            observed: { values: [46, 55, 68, 63] },
            severe: { values: [24, 43, 79, 70] }
        };

        function select(button) {
            const state = states[button.dataset.spread];
            const mean = state.values.reduce(function (sum, value) { return sum + value; }, 0) / state.values.length;
            const tax = state.values.reduce(function (sum, value) {
                return sum + Math.pow((value - mean) / 100, 2);
            }, 0) / state.values.length;
            const averageLoss = Math.pow(1 - mean / 100, 2);
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            dots.forEach(function (dot, index) {
                dot.style.left = state.values[index] + "%";
            });
            centroid.style.left = mean + "%";
            memberLoss.textContent = (averageLoss + tax).toFixed(3);
            centroidLoss.textContent = averageLoss.toFixed(3);
            schemaTax.textContent = tax.toFixed(3);
            demo.dataset.spread = button.dataset.spread;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[1]);
    }

    function initializeSelectionDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-selection]"));
        const grid = demo.querySelector("[data-rep-grid]");
        const summary = demo.querySelector("[data-selection-summary]");
        const brier = demo.querySelector("[data-selection-brier]");
        const risk = demo.querySelector("[data-selection-risk]");
        const riskNote = demo.querySelector("[data-selection-risk-note]");
        const flips = demo.querySelector("[data-selection-flips]");
        const verdict = demo.querySelector("[data-selection-verdict]");
        const selectedIndices = [2, 3, 10, 11];

        function renderGrid(selected) {
            grid.innerHTML = "";
            for (let index = 0; index < 16; index += 1) {
                const switched = selected && selectedIndices.indexOf(index) !== -1;
                const cell = document.createElement("i");
                cell.className = switched ? "d" : "a";
                cell.textContent = switched ? "3" : "0";
                cell.title = isKorean
                    ? "표기 " + (index + 1) + ": 설정 " + (switched ? "3" : "0")
                    : "spelling " + (index + 1) + ": configuration " + (switched ? "3" : "0");
                grid.appendChild(cell);
            }
        }

        function select(button) {
            const selected = button.dataset.selection === "selected";
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            renderGrid(selected);
            summary.textContent = selected
                ? (isKorean ? "특성 순서가 선택 4개를 바꿈" : "feature order switches 4 choices")
                : (isKorean ? "하나의 레시피를 16개 모두에 고정" : "one recipe is frozen across all 16");
            brier.textContent = selected ? "−0.0051" : (isKorean ? "기준" : "reference");
            risk.textContent = selected ? "0.0075" : "0.0032";
            riskNote.textContent = selected
                ? (isKorean ? "흔들림 2.35배 증가" : "2.35× more wobble")
                : (isKorean ? "기준 흔들림" : "baseline wobble");
            flips.textContent = selected ? "16.9%" : "13.8%";
            verdict.innerHTML = selected
                ? (isKorean
                    ? "<b>표기마다 재튜닝:</b> 평균 손실은 개선되지만 동등한 표들의 불일치는 커집니다."
                    : "<b>Retune each:</b> average loss improves, while equivalent tables disagree more.")
                : (isKorean
                    ? "<b>한 번만 선택:</b> 특성 순서가 레시피를 바꿀 수는 없지만, 각 표기에 최적화된 레시피는 아닙니다."
                    : "<b>Choose once:</b> feature order cannot change the recipe, but the recipe is not optimized for every spelling.");
            demo.dataset.selection = button.dataset.selection;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    function initializeWeightDemo(demo) {
        if (!demo) return;

        const buttons = Array.from(demo.querySelectorAll("[data-weight]"));
        const results = Array.from(demo.querySelectorAll("[data-weight-result]"));
        const note = demo.querySelector("[data-weight-note]");
        const states = isKorean ? {
            uniform: { values: [7, 7, 7], note: "균등 가중은 선택이지 자연법칙이 아닙니다." },
            two: { values: [7, 4, 5], note: "적당한 가중치 변화만으로도 세 결론 중 두 개가 약해집니다." },
            four: { values: [6, 2, 4], note: "가중치를 더 크게 바꾸면 해결책이 더 이상 일관되게 유리하지 않습니다." }
        } : {
            uniform: { values: [7, 7, 7], note: "Uniform weighting is a choice, not a law of nature." },
            two: { values: [7, 4, 5], note: "A moderate tilt already weakens two of the three conclusions." },
            four: { values: [6, 2, 4], note: "Under a stronger tilt, the repair is no longer consistently favorable." }
        };

        function select(button) {
            const state = states[button.dataset.weight];
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            results.forEach(function (result, index) {
                result.textContent = state.values[index] + " / 7";
                result.parentElement.classList.toggle("weak", state.values[index] < 5);
            });
            note.textContent = state.note;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    initializeSchemaDemo(document.querySelector("[data-schema-demo]"));
    initializeTaxDemo(document.querySelector("[data-tax-demo]"));
    initializeSelectionDemo(document.querySelector("[data-selection-demo]"));
    initializeWeightDemo(document.querySelector("[data-weight-demo]"));
}());
