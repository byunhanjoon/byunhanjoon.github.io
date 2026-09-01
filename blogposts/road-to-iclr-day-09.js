(function () {
    "use strict";

    document.querySelectorAll("[data-d9-rotate]").forEach(function (lab) {
        const input = lab.querySelector("[data-d9-angle]");
        const paper = lab.querySelector("[data-d9-paper]");
        const angleOutput = lab.querySelector("[data-d9-angle-output]");
        const answer = lab.querySelector("[data-d9-model-answer]");
        const drift = lab.querySelector("[data-d9-drift]");

        function updateRotation() {
            const angle = Number(input.value);
            const magnitude = Math.abs(angle);
            paper.style.setProperty("--angle", angle + "deg");
            drift.style.setProperty("--drift", Math.round(magnitude / 35 * 100) + "%");
            angleOutput.textContent = (angle > 0 ? "+" : "") + angle + "°";
            answer.textContent = magnitude < 2 ? lab.dataset.rest : lab.dataset.drift;
        }

        input.addEventListener("input", updateRotation);
        updateRotation();
    });

    document.querySelectorAll("[data-d9-embedding]").forEach(function (lab) {
        const numberInput = lab.querySelector("[data-d9-number]");
        const basisInput = lab.querySelector("[data-d9-basis]");
        const numberOutput = lab.querySelector("[data-d9-number-output]");
        const numberLabel = lab.querySelector("[data-d9-number-label]");
        const basisOutput = lab.querySelector("[data-d9-basis-output]");
        const answer = lab.querySelector("[data-d9-hidden-answer]");
        const barsRoot = lab.querySelector("[data-d9-code-bars]");
        const bars = Array.from({ length: 16 }, function () {
            const bar = document.createElement("i");
            const fill = document.createElement("span");
            bar.appendChild(fill);
            barsRoot.appendChild(bar);
            return bar;
        });

        function hiddenCode(value, angle) {
            const sigma = 0.13;
            const base = bars.map(function (_, index) {
                const center = index / (bars.length - 1);
                return Math.exp(-0.5 * Math.pow((value - center) / sigma, 2));
            });
            const norm = Math.sqrt(base.reduce(function (sum, item) { return sum + item * item; }, 0));
            const unit = base.map(function (item) { return item / norm; });
            const radians = angle * Math.PI / 180;
            const cosine = Math.cos(radians);
            const sine = Math.sin(radians);
            const rotated = [];

            for (let index = 0; index < unit.length; index += 2) {
                rotated[index] = cosine * unit[index] - sine * unit[index + 1];
                rotated[index + 1] = sine * unit[index] + cosine * unit[index + 1];
            }
            return rotated;
        }

        function updateEmbedding() {
            const value = Number(numberInput.value) / 100;
            const angle = Number(basisInput.value);
            const code = hiddenCode(value, angle);
            const peak = Math.max.apply(null, code.map(Math.abs));

            code.forEach(function (coordinate, index) {
                const magnitude = Math.max(3, Math.abs(coordinate) / peak * 92);
                bars[index].classList.toggle("is-negative", coordinate < 0);
                bars[index].style.setProperty("--height", magnitude + "%");
            });

            const formatted = value.toFixed(2);
            numberOutput.textContent = formatted;
            numberLabel.textContent = formatted;
            basisOutput.textContent = (angle > 0 ? "+" : "") + angle + "°";
            answer.textContent = Math.abs(angle) < 3 ? lab.dataset.rest : lab.dataset.drift;
        }

        numberInput.addEventListener("input", updateEmbedding);
        basisInput.addEventListener("input", updateEmbedding);
        updateEmbedding();
    });

    document.querySelectorAll("[data-d9-dimension]").forEach(function (lab) {
        const buttons = Array.from(lab.querySelectorAll("[data-d9-k]"));
        const dots = lab.querySelector("[data-d9-dimension-dots]");
        const copy = lab.querySelector("[data-d9-dimension-copy]");

        function setDimension(dimension) {
            dots.replaceChildren();
            dots.style.setProperty("--columns", dimension <= 8 ? dimension : 8);
            Array.from({ length: dimension }).forEach(function (_, index) {
                const dot = document.createElement("i");
                dot.style.setProperty("--delay", index * 12 + "ms");
                dots.appendChild(dot);
            });
            copy.textContent = lab.dataset.copyTemplate.replace("{k}", dimension);
            buttons.forEach(function (button) {
                button.setAttribute("aria-pressed", String(Number(button.dataset.d9K) === dimension));
            });
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { setDimension(Number(button.dataset.d9K)); });
        });
        setDimension(16);
    });

    document.querySelectorAll("[data-d9-guard]").forEach(function (lab) {
        const buttons = Array.from(lab.querySelectorAll("[data-d9-guard-case]"));
        const steps = Array.from(lab.querySelectorAll("[data-d9-guard-step]"));
        const output = lab.querySelector("[data-d9-guard-output]");
        const message = lab.querySelector("[data-d9-guard-message]");

        function setCase(example) {
            const selected = example === "clear" ? 75 : 25;
            steps.forEach(function (step) {
                const value = Number(step.dataset.d9GuardStep);
                const status = step.querySelector("[data-d9-guard-status]");
                const accepted = value === selected;
                const rejected = example === "warning" && value > selected;
                step.classList.toggle("is-accepted", accepted);
                step.classList.toggle("is-rejected", rejected);
                step.classList.toggle("is-muted", !accepted && !rejected);
                status.textContent = accepted
                    ? lab.dataset.accepted
                    : rejected
                        ? lab.dataset.rejected
                        : lab.dataset.notNeeded;
            });
            buttons.forEach(function (button) {
                button.setAttribute("aria-pressed", String(button.dataset.d9GuardCase === example));
            });
            output.textContent = selected + "%" + lab.dataset.stableSuffix;
            message.textContent = example === "clear" ? lab.dataset.clearMessage : lab.dataset.warningMessage;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { setCase(button.dataset.d9GuardCase); });
        });
        setCase("clear");
    });
})();
