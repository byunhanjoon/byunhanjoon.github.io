(function () {
    "use strict";

    const korean = (document.documentElement.lang || "en").slice(0, 2) === "ko";

    function formatVector(values) {
        return "[" + values.join(", ") + "]";
    }

    function initializeBasisDemo(demo) {
        if (!demo) {
            return;
        }

        const buttons = Array.from(demo.querySelectorAll("[data-state]"));
        const pleVector = demo.querySelector("[data-ple-vector]");
        const identityVector = demo.querySelector("[data-identity-vector]");
        const pleCode = demo.querySelector("[data-ple-code]");
        const identityCode = demo.querySelector("[data-identity-code]");
        const note = demo.querySelector("[data-basis-note]");

        function cells(container, values) {
            container.innerHTML = "";
            values.forEach(function (value) {
                const cell = document.createElement("span");
                cell.style.setProperty("--activation", value * 100 + "%");
                const bar = document.createElement("i");
                const label = document.createElement("b");
                label.textContent = value;
                cell.appendChild(bar);
                cell.appendChild(label);
                container.appendChild(cell);
            });
        }

        function select(button) {
            const state = Number(button.dataset.state);
            const ple = Array.from({ length: 4 }, function (_, index) {
                return state > index ? 1 : 0;
            });
            const identity = Array.from({ length: 5 }, function (_, index) {
                return state === index ? 1 : 0;
            });

            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            cells(pleVector, ple);
            cells(identityVector, identity);
            pleCode.textContent = formatVector(ple);
            identityCode.textContent = formatVector(identity);
            if (korean) {
                note.textContent = state === 0
                    ? "상태 0은 PLE의 원점이지만, 식별자 기저에서는 여전히 명시적인 좌표를 가집니다."
                    : (state - 1) + "에서 " + state + "로 이동하면 누적 PLE 좌표 하나가 바뀌고, 식별자 기저에서는 활성 좌표 하나가 옮겨갑니다.";
            } else {
                note.textContent = state === 0
                    ? "State 0 is the PLE origin, while identity still gives it an explicit coordinate."
                    : "Moving from " + (state - 1) + " to " + state + " changes one cumulative PLE coordinate—and moves the single active identity coordinate.";
            }
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                select(button);
            });
        });
        select(buttons[2]);
    }

    function mixColor(value) {
        const white = [250, 250, 247];
        const target = value >= 0 ? [35, 122, 120] : [216, 101, 79];
        const amount = Math.min(Math.abs(value), 1) * 0.9;
        return "rgb(" + white.map(function (channel, index) {
            return Math.round(channel + amount * (target[index] - channel));
        }).join(",") + ")";
    }

    function initializeSurfaceDemo(demo) {
        if (!demo) {
            return;
        }

        const buttons = Array.from(demo.querySelectorAll("[data-pattern]"));
        const grid = demo.querySelector("[data-residual-grid]");
        const title = demo.querySelector("[data-surface-title]");
        const formula = demo.querySelector("[data-surface-formula]");
        const note = demo.querySelector("[data-surface-note]");
        const row = [-0.7, -0.25, 0, 0.25, 0.7];
        const column = [-0.45, -0.1, 0, 0.1, 0.45];
        const left = [-1, -0.5, 0, 0.5, 1];
        const right = [1, -1, 0, -1, 1];

        function render(pattern) {
            grid.innerHTML = "";
            const values = [];
            for (let y = 0; y < 5; y += 1) {
                for (let x = 0; x < 5; x += 1) {
                    values.push(pattern === "additive"
                        ? Math.max(-1, Math.min(1, row[y] + column[x]))
                        : left[y] * right[x]);
                }
            }
            values.forEach(function (value) {
                const cell = document.createElement("span");
                cell.style.backgroundColor = mixColor(value);
                cell.title = "residual " + value.toFixed(2);
                grid.appendChild(cell);
            });
            grid.setAttribute(
                "aria-label",
                pattern === "additive"
                    ? (korean ? "행과 열 띠가 나타나는 가산 잔차 표면 예시" : "Illustrative additive residual surface with row and column bands")
                    : (korean ? "행과 열 평균이 0인 순수 상호작용 표면 예시" : "Illustrative pure interaction surface with zero row and column averages")
            );
        }

        function select(button) {
            const pattern = button.dataset.pattern;
            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            if (pattern === "additive") {
                title.textContent = korean ? "두 개의 일차원 효과를 더합니다" : "Add two one-dimensional effects";
                formula.textContent = "r(x, z) = a(x) + b(z)";
                note.textContent = korean
                    ? "행과 열이 패턴을 담습니다. 교차 상태는 필요하지 않습니다."
                    : "Rows and columns carry the pattern. A crossed state is unnecessary.";
            } else {
                title.textContent = korean ? "결합된 나머지만 남깁니다" : "Keep only the joint remainder";
                formula.textContent = "r(x, z) = c(x, z),   E[c|x] = E[c|z] = 0";
                note.textContent = korean
                    ? "어떤 행이나 열도 이 패턴을 단독으로 설명하지 못합니다. 교차 상태가 학습해야 하는 부분입니다."
                    : "No row or column explains the pattern alone. This is the part a crossed state must learn.";
            }
            render(pattern);
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                select(button);
            });
        });
        select(buttons[0]);
    }

    function initializeLocalBasisDemo(demo) {
        if (!demo) {
            return;
        }

        const slider = demo.querySelector("[data-local-slider]");
        const valueOutput = demo.querySelector("[data-local-value]");
        const cumulativeVector = demo.querySelector("[data-cumulative-vector]");
        const localVector = demo.querySelector("[data-local-vector]");
        const cumulativeCode = demo.querySelector("[data-cumulative-code]");
        const localCode = demo.querySelector("[data-local-code]");
        const note = demo.querySelector("[data-local-note]");
        const ramps = 5;

        function rounded(values) {
            return values.map(function (value) {
                return value.toFixed(2);
            });
        }

        function cells(container, values) {
            container.innerHTML = "";
            values.forEach(function (value) {
                const cell = document.createElement("span");
                cell.style.setProperty("--activation", value * 100 + "%");
                const bar = document.createElement("i");
                const label = document.createElement("b");
                label.textContent = value.toFixed(2);
                cell.appendChild(bar);
                cell.appendChild(label);
                container.appendChild(cell);
            });
        }

        function render() {
            const x = Number(slider.value) / 100;
            const width = 1 / ramps;
            const cumulative = Array.from({ length: ramps }, function (_, index) {
                return Math.max(0, Math.min(1, (x - index * width) / width));
            });
            const local = [1 - cumulative[0]];
            for (let index = 1; index < ramps; index += 1) {
                local.push(cumulative[index - 1] - cumulative[index]);
            }

            valueOutput.textContent = x.toFixed(2);
            cells(cumulativeVector, cumulative);
            cells(localVector, local);
            cumulativeCode.textContent = formatVector(rounded(cumulative));
            localCode.textContent = formatVector(rounded(local));
            const cumulativeActive = cumulative.filter(function (value) { return value > 0.001; }).length;
            const localActive = local.filter(function (value) { return value > 0.001; }).length;
            note.textContent = korean
                ? "누적 기저는 " + cumulativeActive + "개 좌표, 국소 기저는 " + localActive + "개 좌표가 활성화됩니다. 표현하는 값은 같습니다."
                : "The cumulative basis activates " + cumulativeActive + " coordinates; the local basis activates " + localActive + ". The represented value is unchanged.";
        }

        slider.addEventListener("input", render);
        render();
    }

    initializeBasisDemo(document.querySelector("[data-basis-demo]"));
    initializeSurfaceDemo(document.querySelector("[data-surface-demo]"));
    initializeLocalBasisDemo(document.querySelector("[data-local-basis-demo]"));
})();
