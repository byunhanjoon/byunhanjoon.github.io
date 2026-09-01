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

    document.querySelectorAll("[data-d9-view]").forEach(function (lab) {
        const buttons = Array.from(lab.querySelectorAll("[data-d9-view-button]"));
        const copy = lab.querySelector("[data-d9-view-copy]");

        function setView(view) {
            const landmarks = view === "landmarks";
            lab.classList.toggle("is-landmarks", landmarks);
            copy.textContent = landmarks ? copy.dataset.landmarkCopy : copy.dataset.axisCopy;
            buttons.forEach(function (button) {
                button.setAttribute("aria-pressed", String(button.dataset.d9ViewButton === view));
            });
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () {
                setView(button.dataset.d9ViewButton);
            });
        });
        setView("axes");
    });

    document.querySelectorAll("[data-d9-blend]").forEach(function (lab) {
        const input = lab.querySelector("[data-d9-blend-input]");
        const rawBar = lab.querySelector("[data-d9-raw-bar]");
        const stableBar = lab.querySelector("[data-d9-stable-bar]");
        const output = lab.querySelector("[data-d9-blend-output]");
        const sensitivity = lab.querySelector("[data-d9-sensitivity]");
        const message = lab.querySelector("[data-d9-blend-message]");

        function updateBlend() {
            const stable = Number(input.value);
            const raw = 100 - stable;
            rawBar.style.width = raw + "%";
            stableBar.style.width = stable + "%";
            rawBar.textContent = raw >= 20 ? rawBar.dataset.label || rawBar.textContent : "";
            stableBar.textContent = stable >= 20 ? stableBar.dataset.label || stableBar.textContent : "";
            output.textContent = stable + "%";
            sensitivity.textContent = raw + "%";
            message.textContent = stable === 0
                ? lab.dataset.msgRaw
                : stable === 100
                    ? lab.dataset.msgStable
                    : lab.dataset.msgBlend;
        }

        rawBar.dataset.label = rawBar.textContent;
        stableBar.dataset.label = stableBar.textContent;
        input.addEventListener("input", updateBlend);
        updateBlend();
    });
})();
