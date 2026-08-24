(function () {
    "use strict";

    function initializeSemanticContinuum(visualization) {
        if (!visualization) {
            return;
        }

        const marker = visualization.querySelector(".semantic-marker");
        const title = visualization.querySelector(".semantic-title");
        const note = visualization.querySelector(".semantic-note");
        const identityBar = visualization.querySelector(".identity-weight");
        const metricBar = visualization.querySelector(".metric-weight");
        const buttons = visualization.querySelectorAll(".feature-tab");

        function selectFeature(button) {
            buttons.forEach(function (candidate) {
                candidate.setAttribute(
                    "aria-selected",
                    candidate === button ? "true" : "false"
                );
            });

            const position = Number(button.dataset.position);
            marker.style.setProperty("--position", position + "%");
            identityBar.style.setProperty("--weight", 100 - position + "%");
            metricBar.style.setProperty("--weight", position + "%");
            title.textContent = button.dataset.title;
            note.textContent = button.dataset.note;
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                selectFeature(button);
            });
        });

        selectFeature(buttons[0]);
    }

    function initializeEncoderDemo(demo) {
        const buttons = demo.querySelectorAll(".encoder-value-button");
        const rawValue = demo.querySelector("[data-raw-value]");
        const pleCells = demo.querySelectorAll("[data-ple-cell]");
        const identityCells = demo.querySelectorAll("[data-identity-cell]");
        const pleCode = demo.querySelector("[data-ple-code]");
        const identityCode = demo.querySelector("[data-identity-code]");
        const note = demo.querySelector("[data-encoder-note]");

        function formatVector(values) {
            return "[" + values.map(function (value) {
                return value === 0.5 ? ".5" : String(value);
            }).join(", ") + "]";
        }

        function update(button) {
            const ple = button.dataset.ple.split(",").map(Number);
            const identity = button.dataset.identity.split(",").map(Number);

            buttons.forEach(function (candidate) {
                candidate.setAttribute(
                    "aria-pressed",
                    candidate === button ? "true" : "false"
                );
            });
            rawValue.textContent = button.dataset.value;
            pleCode.textContent = formatVector(ple);
            identityCode.textContent = formatVector(identity);
            note.textContent = button.dataset.note;

            pleCells.forEach(function (cell, index) {
                cell.style.setProperty("--activation", ple[index] * 100 + "%");
                cell.querySelector("b").textContent = ple[index] === 0.5 ? ".5" : ple[index];
            });
            identityCells.forEach(function (cell, index) {
                cell.style.setProperty("--activation", identity[index] * 100 + "%");
                cell.querySelector("b").textContent = identity[index];
            });
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                update(button);
            });
        });

        update(buttons[1] || buttons[0]);
    }

    initializeSemanticContinuum(document.querySelector(".semantic-viz"));
    document.querySelectorAll("[data-encoder-demo]").forEach(initializeEncoderDemo);
})();
