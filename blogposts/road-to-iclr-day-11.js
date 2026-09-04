(function () {
    "use strict";

    document.querySelectorAll("[data-d11-direction]").forEach(function (lab) {
        const input = lab.querySelector("[data-d11-extra]");
        const output = lab.querySelector("[data-d11-output]");
        const count = lab.querySelector("[data-d11-count]");
        const forward = lab.querySelector("[data-d11-forward]");
        const extras = lab.querySelectorAll("[data-extra-index]");

        function update() {
            const amount = Number(input.value);
            output.textContent = "+" + amount;
            count.textContent = amount;
            forward.textContent = amount === 0 ? lab.dataset.same : lab.dataset.missing;

            extras.forEach(function (slot) {
                const visible = Number(slot.dataset.extraIndex) <= amount;
                slot.classList.toggle("is-visible", visible);
            });
        }

        input.addEventListener("input", update);
        update();
    });
})();
