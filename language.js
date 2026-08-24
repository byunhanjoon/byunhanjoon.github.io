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
        "Manuscripts Under Review": "심사 중인 원고",
        "Byun, Han Joon is a researcher at Seoul National University working on tabular data, time series, finance, machine learning, and optimization.": "Byun, Han Joon은 서울대학교에서 테이블 데이터, 시계열, 금융, 머신러닝, 최적화를 연구하고 있습니다.",
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
