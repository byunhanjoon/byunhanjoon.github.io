(function () {
    "use strict";

    const korean = (document.documentElement.lang || "en").slice(0, 2) === "ko";

    function initializeKappaDemo(demo) {
        if (!demo) {
            return;
        }

        const buttons = Array.from(demo.querySelectorAll("[data-kappa]"));
        const output = demo.querySelector("[data-kappa-output]");
        const note = demo.querySelector("[data-kappa-note]");
        const path = demo.querySelector("[data-step-path]");
        const contours = {
            outer: demo.querySelector('[data-contour="outer"]'),
            middle: demo.querySelector('[data-contour="middle"]'),
            inner: demo.querySelector('[data-contour="inner"]')
        };
        const copy = {
            en: {
                1: "Directions are balanced. A single step size can make useful progress in each one.",
                10: "The valley is becoming oblique. Coordinate-wise scaling only partly follows it.",
                100: "Safe steps across the narrow direction make progress along the valley painfully slow.",
                1000: "The information is unchanged, but finite optimization now faces a long, tightly coupled valley."
            },
            ko: {
                1: "각 방향의 균형이 맞습니다. 하나의 step size로 모든 방향에서 유용하게 전진할 수 있습니다.",
                10: "골짜기가 비스듬해집니다. 좌표별 scale 조정만으로는 일부만 따라갈 수 있습니다.",
                100: "좁은 방향에서 안전한 step은 골짜기의 긴 방향에서 매우 느립니다.",
                1000: "정보는 그대로지만 유한 시간 최적화는 길고 강하게 결합된 골짜기를 만납니다."
            }
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
            [contours.outer, contours.middle, contours.inner].forEach(function (ellipse, index) {
                ellipse.setAttribute("rx", sizes[index][0].toFixed(1));
                ellipse.setAttribute("ry", sizes[index][1].toFixed(1));
            });
            path.setAttribute("d", paths[kappa]);
            output.textContent = kappa.toLocaleString("en-US");
            note.textContent = copy[korean ? "ko" : "en"][kappa];
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    function initializeCanonicalDemo(demo) {
        if (!demo) {
            return;
        }

        const buttons = Array.from(demo.querySelectorAll("[data-basis]"));
        const points = Array.from(demo.querySelectorAll("[data-point]"));
        const note = demo.querySelector("[data-canonical-note]");
        const reference = [
            [58, 35], [42, 47], [67, 53], [34, 31], [52, 66], [24, 69], [73, 72]
        ];
        const canonical = [
            [54, 31], [39, 48], [66, 46], [31, 35], [51, 63], [22, 74], [77, 74]
        ];
        const copy = {
            en: {
                reference: "The coordinates and anchors are in the reference basis.",
                stretched: "The cloud and both anchors moved; the coefficients stayed 2 and −0.5.",
                canonical: "The train-fitted anchor map returns every equivalent basis to the same coefficients."
            },
            ko: {
                reference: "좌표와 anchor가 기준 기저에 있습니다.",
                stretched: "점 구름과 두 anchor는 움직였지만 계수 2와 −0.5는 그대로입니다.",
                canonical: "학습 데이터로 맞춘 anchor map은 모든 동등한 기저를 같은 계수로 돌려놓습니다."
            }
        };

        function transformed(point) {
            const centeredX = point[0] - 50;
            const centeredY = point[1] - 50;
            return [
                Math.max(6, Math.min(94, 50 + 1.72 * centeredX + 0.62 * centeredY)),
                Math.max(8, Math.min(92, 50 + 0.24 * centeredX + 0.55 * centeredY))
            ];
        }

        function select(button) {
            const basis = button.dataset.basis;
            const positions = basis === "canonical"
                ? canonical
                : reference.map(function (point) {
                    return basis === "stretched" ? transformed(point) : point;
                });

            buttons.forEach(function (candidate) {
                candidate.setAttribute("aria-pressed", candidate === button ? "true" : "false");
            });
            points.forEach(function (point, index) {
                point.style.left = positions[index][0] + "%";
                point.style.top = positions[index][1] + "%";
            });
            demo.dataset.activeBasis = basis;
            note.textContent = copy[korean ? "ko" : "en"][basis];
        }

        buttons.forEach(function (button) {
            button.addEventListener("click", function () { select(button); });
        });
        select(buttons[0]);
    }

    initializeKappaDemo(document.querySelector("[data-kappa-demo]"));
    initializeCanonicalDemo(document.querySelector("[data-canonical-demo]"));
})();
