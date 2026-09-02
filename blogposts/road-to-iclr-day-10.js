(function () {
    "use strict";

    document.querySelectorAll("[data-d10-gauge]").forEach(function (lab) {
        const input = lab.querySelector("[data-d10-angle]");
        const plane = lab.querySelector("[data-d10-plane]");
        const output = lab.querySelector("[data-d10-output]");
        const rawStatus = lab.querySelector("[data-d10-raw-status]");
        const fixedStatus = lab.querySelector("[data-d10-fixed-status]");

        function update() {
            const angle = Number(input.value);
            plane.style.setProperty("--turn", angle + "deg");
            output.textContent = (angle > 0 ? "+" : "") + angle + "°";
            rawStatus.style.setProperty("--activity", Math.max(12, Math.abs(angle)) + "%");
            rawStatus.textContent = Math.abs(angle) < 2 ? lab.dataset.fixed : lab.dataset.free;
            fixedStatus.textContent = lab.dataset.fixed;
        }

        input.addEventListener("input", update);
        update();
    });
})();
