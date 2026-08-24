(function () {
    "use strict";

    const visualization = document.querySelector(".semantic-viz");
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
            candidate.setAttribute("aria-selected", candidate === button ? "true" : "false");
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
})();
