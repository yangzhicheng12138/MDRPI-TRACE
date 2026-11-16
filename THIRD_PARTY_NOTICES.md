# Third-party components

Free public web discovery uses the unmodified DDGS package (MIT), version 9.16.0,
installed via requirements.txt. It does not require a commercial search API key.
Project: https://github.com/deedy5/ddgs
License: https://github.com/deedy5/ddgs/blob/main/LICENSE.md

The Brave request/result mapping in `app/web_search.py` is adapted from LangChain Community BraveSearchWrapper.
Source: https://github.com/langchain-ai/langchain-community/blob/main/libs/community/langchain_community/utilities/brave_search.py
Retrieved: 2026-09-28. The project transport and snapshot/replay integration are retained.

MIT License

Copyright (c) 2024 LangChain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


HTML body extraction uses the unmodified Trafilatura package (Apache-2.0), installed via requirements.txt.
Project: https://github.com/adbar/trafilatura
License: https://github.com/adbar/trafilatura/blob/master/LICENSE


Optional dynamic-page rendering uses the unmodified Crawl4AI package (Apache-2.0),
version 0.9.4, in a separate CPU environment installed from requirements-web.txt.
Project: https://github.com/unclecode/crawl4ai
License: https://github.com/unclecode/crawl4ai/blob/main/LICENSE
The project calls its AsyncWebCrawler API; no search engine or LLM is supplied by
this reader. Browser automation is provided by its Playwright dependency.
Project: https://github.com/microsoft/playwright-python (Apache-2.0).

The optional interactive modes use the unmodified Chainlit package (Apache-2.0)
and LangGraph / langgraph-checkpoint-sqlite packages (MIT), installed in the
separate environment specified by requirements.txt. Their distributed
packages retain the upstream licenses.
Projects: https://github.com/Chainlit/chainlit and https://github.com/langchain-ai/langgraph
Licenses: https://github.com/Chainlit/chainlit/blob/main/LICENSE and
https://github.com/langchain-ai/langgraph/blob/main/LICENSE
OATutor (https://github.com/CAHLR/OATutor) informed progressive tutoring hints.
The small dependency-unlock function in modes/vendor/oatutor_hints.py is adapted
from revision 939eb0e30d8ae2389c8b096c8cd522713a79090e under MIT; copyright and
the complete license are retained in modes/vendor/OATUTOR_LICENSE. No upstream
mathematics curriculum or mastery parameters are used for nursing assessment.

Teaching review scheduling calls the unmodified fsrs Python package, version
6.3.2, under MIT, installed only in the interactive-modes CPU environment.
Project: https://github.com/open-spaced-repetition/py-fsrs
Copyright (c) 2022 Open Spaced Repetition.
License: modes/vendor/FSRS_LICENSE and the distributed package's license.

## Chat frontend browser libraries

The local workspace uses Marked 18.0.14 (MIT), DOMPurify 3.4.16
(Apache-2.0 option of its dual license), and highlight.js CDN assets 11.12.0
(BSD-3-Clause). Their unmodified browser distributions and complete license
texts are in `modes/frontend/vendor/`. `manifest.json` records the official
npm tarballs, verified SHA-512 integrity and installed file SHA-256 hashes.
No CDN connection is required to render the interface.

Sources: https://github.com/markedjs/marked,
https://github.com/cure53/DOMPurify,
https://github.com/highlightjs/highlight.js.

Open WebUI and LibreChat are interaction/design references for conversation
navigation, input, and themes. Their application code, logos, and backends
were not copied into the new frontend. The workspace layout was written for
this project. The nursing-hand/ribbon T logo was supplied by the project owner;
the browser SVG asset embeds that image. It is not an upstream project's logo.
Review ratings are explicitly selected by the learner. Default scheduler weights
are not reported as validated clinical skill estimates. The optional optimizer
and its GPU dependencies are not installed.

## Document processing and teaching content

PyMuPDF is an external dependency available under GNU AGPL v3 or a commercial
license from Artifex. The MIT license for this project's original code does not
relicense PyMuPDF. Deployers and distributors must account for the applicable
PyMuPDF license when distributing or operating a combined application.
Project and licensing: https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright

Other document/image dependencies include pypdf (BSD-3-Clause), python-docx
(MIT), and Pillow (HPND). Their installed distributions retain their notices.

The Open RN teaching content is attributed in data/teaching_bank/openrn.json.
The source is CC BY 4.0; the teaching adaptations distributed here are labeled
CC BY-SA 4.0. Project-created teaching examples retain their per-item content
license. The MIT code license does not override these content licenses.
See data/teaching_bank/README.md for attribution and scope.
