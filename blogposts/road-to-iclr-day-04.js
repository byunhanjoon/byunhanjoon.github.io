(() => {
    "use strict";

    const svgNS = "http://www.w3.org/2000/svg";

    const geometries = {
        ordered: {
            title: "Ordered values live on a path",
            copy: "Nearby values may share statistical strength. The two ends stay far apart.",
            rule: "1—2—3—4—5—6—7",
            description: "Seven values connected along a path.",
            points: [[70, 130], [153, 130], [236, 130], [320, 130], [404, 130], [487, 130], [570, 130]],
            labels: ["1", "2", "3", "4", "5", "6", "7"],
            edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]],
            labelOffset: [0, 51]
        },
        cyclic: {
            title: "Cyclic values close the loop",
            copy: "The last value returns to the first. Midnight is close to 11 p.m., not at the opposite end of time.",
            rule: "23 ↔ 00 as well as 00 ↔ 01",
            description: "Eight hour values connected in a cycle.",
            points: [[320, 37], [432, 70], [478, 130], [432, 190], [320, 223], [208, 190], [162, 130], [208, 70]],
            labels: ["00", "03", "06", "09", "12", "15", "18", "21"],
            edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]],
            labelOffset: [0, 5]
        },
        nominal: {
            title: "Nominal values refuse invented neighbors",
            copy: "Codes remain distinct names. The model may learn similarities, but the schema does not smuggle in an arbitrary path.",
            rule: "A, B, C, D … are distinct; no edge is assumed",
            description: "Seven category labels shown as disconnected points.",
            points: [[95, 72], [241, 54], [407, 75], [543, 55], [157, 191], [328, 204], [504, 180]],
            labels: ["A", "B", "C", "D", "E", "F", "G"],
            edges: [],
            labelOffset: [0, 7]
        }
    };

    const createSvgElement = (tag, attributes = {}) => {
        const element = document.createElementNS(svgNS, tag);
        Object.entries(attributes).forEach(([name, value]) => element.setAttribute(name, String(value)));
        return element;
    };

    const domainDemo = document.querySelector("[data-domain-demo]");
    if (domainDemo) {
        const buttons = [...domainDemo.querySelectorAll("[data-domain]")];
        const title = domainDemo.querySelector("[data-domain-title]");
        const copy = domainDemo.querySelector("[data-domain-copy]");
        const rule = domainDemo.querySelector("[data-domain-rule]");
        const edgesGroup = domainDemo.querySelector("[data-domain-edges]");
        const nodesGroup = domainDemo.querySelector("[data-domain-nodes]");
        const labelsGroup = domainDemo.querySelector("[data-domain-labels]");
        const svgTitle = domainDemo.querySelector("#d4-domain-svg-title");
        const svgDescription = domainDemo.querySelector("#d4-domain-svg-description");

        const renderGeometry = (key) => {
            const geometry = geometries[key];
            if (!geometry) return;

            title.textContent = geometry.title;
            copy.textContent = geometry.copy;
            rule.textContent = geometry.rule;
            svgTitle.textContent = geometry.title;
            svgDescription.textContent = geometry.description;
            edgesGroup.replaceChildren();
            nodesGroup.replaceChildren();
            labelsGroup.replaceChildren();

            geometry.edges.forEach(([from, to]) => {
                const [x1, y1] = geometry.points[from];
                const [x2, y2] = geometry.points[to];
                edgesGroup.append(createSvgElement("line", {x1, y1, x2, y2}));
            });

            geometry.points.forEach(([cx, cy], index) => {
                nodesGroup.append(createSvgElement("circle", {cx, cy, r: 19}));
                const text = createSvgElement("text", {
                    x: cx + geometry.labelOffset[0],
                    y: cy + geometry.labelOffset[1]
                });
                text.textContent = geometry.labels[index];
                labelsGroup.append(text);
            });

            buttons.forEach((button) => {
                button.setAttribute("aria-pressed", String(button.dataset.domain === key));
            });
        };

        buttons.forEach((button) => {
            button.addEventListener("click", () => renderGeometry(button.dataset.domain));
        });
    }

    const smoothDemo = document.querySelector("[data-smooth-demo]");
    if (smoothDemo) {
        const range = smoothDemo.querySelector("[data-tau-range]");
        const output = smoothDemo.querySelector("[data-tau-output]");
        const curve = smoothDemo.querySelector("[data-tau-curve]");
        const title = smoothDemo.querySelector("[data-smooth-title]");
        const note = smoothDemo.querySelector("[data-smooth-note]");
        const xs = [65, 133, 201, 269, 337, 405, 473, 541, 575];
        const observed = [210, 92, 212, 175, 58, 203, 187, 88, 120];
        const smooth = [168, 158, 145, 132, 124, 128, 139, 151, 158];

        const pathFromPoints = (points) => {
            let path = `M${points[0][0]} ${points[0][1]}`;
            for (let index = 1; index < points.length; index += 1) {
                const [x0, y0] = points[index - 1];
                const [x1, y1] = points[index];
                const midpoint = (x0 + x1) / 2;
                path += ` C${midpoint} ${y0} ${midpoint} ${y1} ${x1} ${y1}`;
            }
            return path;
        };

        const renderSmoothness = () => {
            const amount = Number(range.value) / 100;
            const tau = amount * 3;
            const blend = 1 - Math.exp(-2.4 * amount);
            const points = xs.map((x, index) => [x, observed[index] * (1 - blend) + smooth[index] * blend]);
            curve.setAttribute("d", pathFromPoints(points));
            output.value = `τ = ${tau.toFixed(1)}`;
            output.textContent = output.value;
            range.style.setProperty("--d4-range", `${range.value}%`);

            if (amount < 0.18) {
                title.textContent = "Mostly follow the observed pattern";
                note.textContent = "Small local wiggles remain cheap; the declared neighborhood is only a gentle prior.";
            } else if (amount < 0.62) {
                title.textContent = "Prefer changes that vary gradually";
                note.textContent = "The model can still bend, but rapid neighbor-to-neighbor oscillations cost more.";
            } else {
                title.textContent = "Trust the declared geometry strongly";
                note.textContent = "Only broad, smooth variation stays cheap. Too much trust can erase a real sharp effect.";
            }
        };

        range.addEventListener("input", renderSmoothness);
        renderSmoothness();
    }
})();
