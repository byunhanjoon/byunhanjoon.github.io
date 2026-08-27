(() => {
    "use strict";

    const svgNS = "http://www.w3.org/2000/svg";
    const isKorean = document.documentElement.lang.slice(0, 2) === "ko";

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

    const koreanGeometryText = {
        ordered: {
            title: "순서형 값은 경로 위에 놓입니다",
            copy: "가까운 값들은 통계적 정보를 공유할 수 있습니다. 양 끝은 여전히 멀리 떨어져 있습니다.",
            rule: "1—2—3—4—5—6—7",
            description: "경로를 따라 연결된 일곱 개의 값."
        },
        cyclic: {
            title: "순환형 값은 고리를 닫습니다",
            copy: "마지막 값은 첫 값으로 돌아옵니다. 자정은 시간의 반대편이 아니라 오후 11시 가까이에 있습니다.",
            rule: "00 ↔ 01뿐 아니라 23 ↔ 00",
            description: "순환 구조로 연결된 여덟 개의 시각 값."
        },
        nominal: {
            title: "명목형 값에는 이웃을 지어내지 않습니다",
            copy: "코드는 서로 다른 이름으로 남습니다. 모델은 유사성을 학습할 수 있지만, 스키마가 임의의 경로를 몰래 주입하지는 않습니다.",
            rule: "A, B, C, D …는 서로 다르며, 어떤 변도 가정하지 않음",
            description: "연결되지 않은 점으로 표시된 일곱 개의 범주 레이블."
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
            const words = isKorean ? koreanGeometryText[key] : geometry;

            title.textContent = words.title;
            copy.textContent = words.copy;
            rule.textContent = words.rule;
            svgTitle.textContent = words.title;
            svgDescription.textContent = words.description;
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
                title.textContent = isKorean ? "관측된 패턴을 주로 따릅니다" : "Mostly follow the observed pattern";
                note.textContent = isKorean ? "작은 국소 요동의 비용은 여전히 낮습니다. 선언한 이웃 관계는 완만한 사전분포로만 작용합니다." : "Small local wiggles remain cheap; the declared neighborhood is only a gentle prior.";
            } else if (amount < 0.62) {
                title.textContent = isKorean ? "서서히 변하는 함수를 선호합니다" : "Prefer changes that vary gradually";
                note.textContent = isKorean ? "모델은 여전히 굽힐 수 있지만, 이웃 사이에서 빠르게 진동하는 변화의 비용이 더 커집니다." : "The model can still bend, but rapid neighbor-to-neighbor oscillations cost more.";
            } else {
                title.textContent = isKorean ? "선언한 기하를 강하게 신뢰합니다" : "Trust the declared geometry strongly";
                note.textContent = isKorean ? "넓고 매끄러운 변화만 낮은 비용을 유지합니다. 지나친 신뢰는 실제로 존재하는 급격한 효과를 지울 수 있습니다." : "Only broad, smooth variation stays cheap. Too much trust can erase a real sharp effect.";
            }
        };

        range.addEventListener("input", renderSmoothness);
        renderSmoothness();
    }
})();
