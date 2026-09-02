(function () {
    "use strict";

    const storageKey = "site-language";
    const koreanText = {
        "Home": "홈",
        "Blog": "블로그",
        "About": "소개",
        "Education": "학력",
        "Experiences": "경력",
        "Teaching": "강의",
        "Teaching (TA)": "강의 조교",
        "Publications": "연구 실적",
        "Research Profile": "연구 소개",
        "Blog Posts": "블로그 글",
        "Posts coming soon.": "게시물이 곧 공개됩니다.",
        "When “Numerical” and “Categorical” Aren’t Types": "“수치형”과 “범주형”이 타입이 아닐 때",
        "Road to ICLR · Day 1 / 30": "Road to ICLR · 1일차 / 30",
        "August 24, 2026": "2026년 8월 24일",
        "A column’s storage type is not its learning semantics. A visual hypothesis about multi-view preprocessing for mixed tabular features.": "열의 저장 타입은 학습 의미론이 아닙니다. 혼합형 정형 특성을 위한 다중 관점 전처리에 관한 시각적 가설입니다.",
        "Read Day 1 →": "1일차 읽기 →",
        "Same Information, Different Path": "같은 정보, 다른 경로",
        "Road to ICLR · Day 2 / 30": "Road to ICLR · 2일차 / 30",
        "August 25, 2026": "2026년 8월 25일",
        "Why an exact-state view can help without adding information—and why the broader numerical-encoder hypothesis failed.": "정확한 상태 관점은 새 정보를 더하지 않고도 왜 도움이 되는지, 그리고 더 넓은 수치 인코더 가설은 왜 실패했는지 살펴봅니다.",
        "Why information-equivalent numerical coordinates can change optimization, predictions, and useful ensembles.": "정보가 동등한 수치 좌표가 최적화와 예측, 유용한 앙상블을 어떻게 바꿀 수 있는지 살펴봅니다.",
        "Read Day 2 →": "2일차 읽기 →",
        "Same Information, Different Difficulty": "같은 정보, 다른 난이도",
        "Road to ICLR · Day 3 / 30": "Road to ICLR · 3일차 / 30",
        "August 26, 2026": "2026년 8월 26일",
        "Why one semantic table can become many learning problems—and how schema risk, initialization, and optimizer geometry connect.": "하나의 의미적 표가 여러 학습 문제로 바뀌는 이유와 스키마 위험, 초기화, 최적화 기하가 어떻게 이어지는지 살펴봅니다.",
        "Read Day 3 →": "3일차 읽기 →",
        "Same Table, Different Views": "같은 표, 다른 관점",
        "Road to ICLR · Day 4 / 30": "Road to ICLR · 4일차 / 30",
        "August 27, 2026": "2026년 8월 27일",
        "Why a matched-compute ensemble benefits when one model reads the same table through a different numerical coordinate system.": "같은 계산량에서 한 모델이 동일한 표를 다른 수치 좌표계로 읽을 때 앙상블이 왜 더 좋아지는지 살펴봅니다.",
        "Read Day 4 →": "4일차 읽기 →",
        "Average the Orbit, Not the Accident": "우연이 아니라 궤도를 평균내기",
        "Road to ICLR · Day 5 / 30": "Road to ICLR · 5일차 / 30",
        "August 28, 2026": "2026년 8월 28일",
        "Why harmless schema rewrites and training seeds should be averaged together—and how balancing every pair gets closer to the full average with fewer training runs.": "의미를 보존하는 스키마 재표기와 학습 시드를 함께 평균내야 하는 이유와 모든 쌍을 균형화해 더 적은 학습으로 전체 평균에 가까워지는 방법을 살펴봅니다.",
        "Read Day 5 →": "5일차 읽기 →",
        "Names Shouldn't Matter. Neighbors Should.": "이름은 중요하지 않다. 이웃은 중요하다.",
        "Road to ICLR · Day 6 / 30": "Road to ICLR · 6일차 / 30",
        "August 29, 2026": "2026년 8월 29일",
        "Why arbitrary category IDs should be ignored, while real neighborhoods can help a model predict states missing from training.": "임의적인 범주 ID는 무시해야 하지만 실제 이웃 관계는 학습에서 빠진 상태를 예측하는 데 왜 도움이 되는지 살펴봅니다.",
        "Read Day 6 →": "6일차 읽기 →",
        "The Average Is Part of the Model": "평균은 모델의 일부다",
        "Road to ICLR · Day 7 / 30": "Road to ICLR · 7일차 / 30",
        "August 30, 2026": "2026년 8월 30일",
        "Why an ensemble must declare what it averages—and why balancing schema and training randomness works only when the two are coupled.": "앙상블이 무엇을 평균내는지 밝혀야 하는 이유와 스키마와 학습 무작위성을 함께 결합할 때만 균형화가 작동하는 이유를 살펴봅니다.",
        "Read Day 7 →": "7일차 읽기 →",
        "Better Neighbors Don't Always Make a Better Crowd": "더 나은 이웃이 항상 더 나은 무리를 만들지는 않는다",
        "Road to ICLR · Day 8 / 30": "Road to ICLR · 8일차 / 30",
        "August 31, 2026": "2026년 8월 31일",
        "Why scoring retrieved neighbors one at a time can misjudge the prediction they make together—and what a failed reliability method taught us about aggregate risk.": "검색된 이웃을 하나씩 평가하면 함께 만드는 예측을 왜 잘못 판단할 수 있는지, 실패한 신뢰도 방법이 집계 위험에 관해 무엇을 가르쳐 주었는지 살펴봅니다.",
        "Read Day 8 →": "8일차 읽기 →",
        "The Model Built Its Own Grid—and Then Trusted It": "모델은 스스로 모눈을 만들고, 그 모눈을 믿었다",
        "Road to ICLR · Day 9 / 30": "Road to ICLR · 9일차 / 30",
        "September 1, 2026": "2026년 9월 1일",
        "An interactive visual essay: watch one number become sixteen hidden coordinates, then see why stability needs a safety brake.": "숫자 하나가 열여섯 개 숨은 좌표가 되는 과정을 보고, 안정성에도 안전 브레이크가 필요한 이유를 살펴보는 인터랙티브 시각 에세이입니다.",
        "Read Day 9 →": "9일차 읽기 →",
        "Some Axes Should Matter. Others Should Disappear.": "어떤 축은 중요해야 하고, 어떤 축은 사라져야 한다.",
        "Road to ICLR · Day 10 / 30": "Road to ICLR · 10일차 / 30",
        "September 2, 2026": "2026년 9월 2일",
        "Why the right fix for a hidden coordinate grid was not to erase it, but to return it to the encoder's native frame.": "숨은 좌표 모눈을 지우는 대신 인코더 본래의 좌표틀로 되돌리는 것이 왜 올바른 해결책이었는지 살펴봅니다.",
        "Read Day 10 →": "10일차 읽기 →",
        "Manuscripts Under Review": "심사 중인 원고",
        "Byun, Han Joon is a researcher at Seoul National University working on tabular data, time series, finance, machine learning, and optimization.": "변한준은 정형 데이터, 시계열, 금융, 머신러닝, 최적화를 연구하는 서울대학교 연구자입니다.",
        "I am a researcher working on tabular data, time series, and finance.": "저는 테이블 데이터, 시계열, 금융을 연구하고 있습니다.",
        "PhD in Computer Science and Engineering": "컴퓨터공학 박사",
        "MS in Computer Science and Engineering": "컴퓨터공학 석사",
        "BS in Mathematics": "수학 학사",
        "Seoul National University": "서울대학교",
        "New York University": "뉴욕대학교",
        "Exp. 2027": "2027년 졸업 예정",
        "Agency for Defense Development (Joint Project)": "국방과학연구소 (공동 연구)",
        "PFCT (Joint Project)": "PFCT (공동 연구)",
        "Think Pool (Joint Project)": "Think Pool (공동 연구)",
        "ROKAF Interpretation Officer": "대한민국 공군 통역장교",
        "Data Structure": "자료구조",
        "Algorithm": "알고리즘",
        "Equal contribution.": "공동 기여.",
        "Submitted to ICAIF 2026 — Under review": "ICAIF 2026 투고 — 심사 중",
        "Submitted to Transactions on Machine Learning Research (TMLR) — Under review": "Transactions on Machine Learning Research (TMLR) 투고 — 심사 중",
        "Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '26)": "Genetic and Evolutionary Computation Conference (GECCO '26) 논문집",
        "Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '26 Companion)": "Genetic and Evolutionary Computation Conference Companion (GECCO '26 Companion) 논문집",
        "Proceedings of the Genetic and Evolutionary Computation Companion (GECCO '23 Companion)": "Genetic and Evolutionary Computation Companion (GECCO '23 Companion) 논문집",
        "arXiv preprint": "arXiv 프리프린트",
        "I studied Mathematics at New York University, where I built a rigorous foundation in mathematical thinking. Alongside my undergraduate coursework, I sat in on graduate courses including Probability Theory, Statistics, and Stochastic Calculus. Seeing how these ideas could describe uncertainty, systems, and real-world decisions made me increasingly fascinated by the applications of mathematics.": "뉴욕대학교에서 수학을 전공하며 엄밀한 수학적 사고의 기초를 다졌습니다. 학부 과정과 함께 확률론, 통계학, 확률미적분학 등의 대학원 수업도 청강했습니다. 이러한 개념들이 불확실성과 시스템, 현실의 의사결정을 설명하는 방식을 접하면서 수학의 응용에 더욱 큰 관심을 갖게 되었습니다.",
        "That interest led me to Computer Science at Seoul National University, where I joined the Optimization Lab. There, I have been able to study and research a wide range of mathematical applications, with particular focus on genetic algorithms, neural networks, and financial engineering. This path has let me connect theoretical tools with practical problems in optimization, machine learning, and finance.": "이 관심을 바탕으로 서울대학교 컴퓨터공학부 최적화 연구실에 합류했습니다. 이곳에서 유전 알고리즘, 신경망, 금융공학을 중심으로 다양한 수학적 응용을 연구해 왔습니다. 이를 통해 이론적 도구를 최적화, 머신러닝, 금융의 실제 문제와 연결하고 있습니다.",
        "During my service in the Republic of Korea Air Force as an interpretation officer, I had time to reflect deeply on my career path. I decided to build expertise at the intersection of computer science and finance, combining analytical rigor with work that can make a tangible difference.": "대한민국 공군 통역장교로 복무하며 진로를 깊이 고민했고, 분석적 엄밀함과 실질적인 가치를 함께 추구할 수 있는 컴퓨터공학과 금융의 접점에서 전문성을 쌓기로 했습니다.",
        "I then joined a university lab focused on finance and AI, where I have taken on projects closely connected to that goal. At Think Pool, I developed AI-driven factors for finance. With PFCT, I worked on a default-prediction model using tabular transformers. At the Agency for Defense Development, I contributed to interpretable-model research. Together, these experiences have strengthened my interest in building reliable, effective, and understandable AI for financial and tabular data.": "이후 금융과 AI를 연구하는 대학 연구실에 합류해 목표와 밀접한 프로젝트들을 수행했습니다. Think Pool에서는 AI 기반 금융 팩터를 개발했고, PFCT에서는 테이블 트랜스포머를 활용한 부도 예측 모델을 연구했습니다. 국방과학연구소에서는 해석 가능한 모델 연구에 참여했습니다. 이러한 경험을 통해 금융 및 테이블 데이터를 위한 신뢰성 있고 효과적이며 이해 가능한 AI를 개발하는 데 관심을 넓혀 왔습니다.",
        "As Head TA for Data Structure and Algorithm courses, I help prepare course materials and make sure that students have the support they need throughout the semester. I aim to be approachable and available whenever students need help working through a concept or assignment.": "자료구조와 알고리즘 과목의 수석 조교로서 강의 자료를 준비하고 학생들이 학기 동안 필요한 지원을 받을 수 있도록 돕고 있습니다. 학생들이 개념이나 과제를 이해하는 데 도움이 필요할 때 편하게 질문할 수 있는 조교가 되고자 합니다.",
        "My responsibilities include preparing relevant course content, answering student questions, coordinating with the teaching team, and grading assignments and exams. I enjoy helping students develop confidence in the core problem-solving skills that these subjects require.": "주요 업무는 강의 콘텐츠 준비, 학생 질문 응대, 조교진 협업, 과제 및 시험 채점입니다. 학생들이 과목에서 요구하는 핵심 문제 해결 능력에 자신감을 갖도록 돕는 일에 보람을 느낍니다.",
        "My research began with pruning and transfer learning for convolutional and artificial neural networks, which gave me a stronger practical understanding of how neural networks learn, adapt, and can be made more efficient.": "연구의 출발점은 합성곱 신경망과 인공 신경망의 가지치기 및 전이학습이었습니다. 이를 통해 신경망이 학습하고 적응하는 방식과 효율을 높이는 방법을 실질적으로 이해하게 되었습니다.",
        "I then helped develop a bespoke Transformer for the Traveling Salesperson Problem, applying language-modeling ideas to combinatorial optimization. Building on those foundations, my recent work brings neural-network and optimization methods to financial and tabular domains, with an emphasis on improving performance, efficiency, and interpretability.": "이후 언어 모델링 개념을 조합 최적화에 적용해 외판원 문제를 위한 트랜스포머 개발에 참여했습니다. 최근에는 이러한 기반을 바탕으로 신경망과 최적화 기법을 금융 및 테이블 데이터 영역에 적용하며 성능, 효율성, 해석 가능성을 개선하는 데 집중하고 있습니다.",
        "© 2026 Han Joon Byun. All rights reserved.": "© 2026 Han Joon Byun. 모든 권리 보유."
    };

    const koreanTitles = {
        "About - Han Joon Byun": "소개 - Han Joon Byun",
        "Education - Han Joon Byun": "학력 - Han Joon Byun",
        "Experiences - Han Joon Byun": "경력 - Han Joon Byun",
        "Teaching - Han Joon Byun": "강의 - Han Joon Byun",
        "Publications - Han Joon Byun": "연구 실적 - Han Joon Byun",
        "Blog - Han Joon Byun": "블로그 - Han Joon Byun"
    };

    const originalTitle = document.title;
    const sourceLanguage = (document.documentElement.lang || "en").slice(0, 2);
    const translatedNodes = [];

    function preserveWhitespace(original, replacement) {
        const leading = original.match(/^\s*/)[0];
        const trailing = original.match(/\s*$/)[0];
        return leading + replacement + trailing;
    }

    function collectTranslatableText() {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let node;

        while ((node = walker.nextNode())) {
            if (["SCRIPT", "STYLE", "CODE", "PRE"].includes(node.parentElement.tagName)) {
                continue;
            }

            const key = node.nodeValue.trim();
            if (koreanText[key]) {
                translatedNodes.push({ node: node, original: node.nodeValue, key: key });
            }
        }
    }

    function setLanguage(language, remember) {
        const useKorean = language === "ko";
        const alternate = document.querySelector(
            'link[rel="alternate"][hreflang="' + language + '"]'
        );

        if (language !== sourceLanguage && alternate) {
            if (remember) {
                try {
                    window.localStorage.setItem(storageKey, language);
                } catch (error) {
                    // Continue to the localized page without persistence.
                }
            }
            window.location.assign(alternate.href);
            return;
        }

        translatedNodes.forEach(function (item) {
            item.node.nodeValue = useKorean
                ? preserveWhitespace(item.original, koreanText[item.key])
                : item.original;
        });

        document.documentElement.lang = useKorean ? "ko" : "en";
        document.title = useKorean && koreanTitles[originalTitle]
            ? koreanTitles[originalTitle]
            : originalTitle;

        const select = document.querySelector(".language-select");
        if (select) {
            select.value = useKorean ? "ko" : "en";
            select.setAttribute("aria-label", useKorean ? "언어 선택" : "Select language");
        }

        if (remember) {
            try {
                window.localStorage.setItem(storageKey, useKorean ? "ko" : "en");
            } catch (error) {
                // Continue without persistence when storage is unavailable.
            }
        }
    }

    function initialLanguage() {
        try {
            const stored = window.localStorage.getItem(storageKey);
            if (stored === "ko" || stored === "en") {
                return stored;
            }
        } catch (error) {
            // Fall through to location-based detection.
        }

        if (sourceLanguage === "ko") {
            return "ko";
        }

        try {
            if (Intl.DateTimeFormat().resolvedOptions().timeZone === "Asia/Seoul") {
                return "ko";
            }
        } catch (error) {
            // Fall through to the global English default.
        }

        return "en";
    }

    function addLanguageSelector() {
        const navLinks = document.querySelector(".nav-links");
        if (!navLinks) {
            return;
        }

        const label = document.createElement("label");
        label.className = "language-selector";
        label.title = "Language";
        label.innerHTML = [
            '<i class="fas fa-globe" aria-hidden="true"></i>',
            '<select class="language-select" aria-label="Select language">',
            '<option value="en">English</option>',
            '<option value="ko">한국어</option>',
            "</select>"
        ].join("");

        navLinks.appendChild(label);
        label.querySelector("select").addEventListener("change", function (event) {
            setLanguage(event.target.value, true);
        });
    }

    addLanguageSelector();
    collectTranslatableText();
    setLanguage(initialLanguage(), false);
})();
