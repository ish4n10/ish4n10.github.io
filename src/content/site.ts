export const NAV = [
  { href: "#about", label: "about", addr: "0x01" },
  { href: "#work", label: "work", addr: "0x02" },
  { href: "#projects", label: "projects", addr: "0x03" },
  { href: "#writing", label: "writing", addr: "0x04" },
  { href: "#contact", label: "contact", addr: "0x05" },
];

export const HERO = {
  header: "// ELF header · entry = _start",

  title: [
    "Backend.",
    "Machine Learning.",
    "Reliable Systems.",
  ],

  subtitle:
    "Software engineer building distributed backend systems, AI infrastructure, and developer tools. Interested in AI Inference Engineering and Low-Level Engineering.",

  readelf: [
    ["Class", "ELF64"],
    ["Machine", "human/ish4n10"],
    ["Type", "EXEC"],
    ["Entry", "0x00001000  _start"],
  ],

  ps: [
    ["currently", "Fullstack Developer Intern @ Actionss"],
    ["building", "Agentic systems & backend infrastructure"],
    ["reading", "AI Inference Engineering"],
  ],
};

export const ABOUT = {
  index: "0x01",
  section: ".about",
  symbol: "describe_self()",

  title: "About",

  paragraphs: [
    "I'm a software engineer focused on backend infrastructure, distributed systems, and applied machine learning. Most of my recent work involves backed, agentic ai and developer tooling.",

    "Outside production engineering, I enjoy building projects from first principles. I like understanding how systems work internally—from transformers and storage engines to symbolic execution and operating system fundamentals."
  ],

  segments: [
    [".focus", "Backend + AI"],
    [".languages", "Go · Python · TypeScript · C++"],
    [".systems", "Distributed Systems"],
    [".interest", "ML Inference · Program Analysis and Reverse Engineering"],
  ],
};


export const EXPERIENCE = [
  {
    period: "Feb 2026 — Present",
    offset: "0x0300",

    symbol: "actonss_backend",

    role: "Fullstack Developer Intern",

    org: "Actionss",

    notes:
      "Building AI-powered recruiting products including LLM-assisted resume coaching, autonomous application agents, and large-scale job recommendation systems.",

    stack: [
      "typescript",
      "nestjs",
      "playwright",
      "llm",
      "websocket",
      "postgres"
    ],
  },

  {
    period: "Dec 2023 — Aug 2025",

    offset: "0x0210",

    symbol: "astroshrine_backend",

    role: "Software Developer Intern",

    org: "Astroshrine",

    notes:
      "Developed scalable backend services, real-time communication systems, retrieval-augmented generation pipelines, payment infrastructure, and asynchronous AWS workflows.",

    stack: [
      "nodejs",
      "aws",
      "mongodb",
      "redis",
      "rag",
      "docker"
    ],
  },
];

export const WORK_HEADER = {
  index: "0x02",
  section: ".history",
  symbol: "trace_calls()",

  title: "Experience",

  subtitle:
    "Professional work building backend services, AI products, and infrastructure.",
};

export const PROJECTS = [
  {
    name: "PocketTransformer",

    offset: "0x00D0",

    arch: "PyTorch",

    tag: "Machine Learning",

    blurb:
      "42M parameter LLaMA-style transformer implemented from scratch with RMSNorm, RoPE, grouped-query attention, ReLU², and custom training pipeline.",

    href: "https://github.com/ish4n10/pocket-transformer",
  },

  {
    name: "MiniatureDB",

    offset: "0x00C0",

    arch: "Go",

    tag: "Storage Engine",

    blurb:
      "Embedded SQL database built on a custom B-tree storage engine with caching, recursive page splitting, and linked-leaf range scans.",

    href: "https://github.com/ish4n10/miniature-db",
  },

  {
    name: "StaleGuard",

    offset: "0x00B0",

    arch: "Python",

    tag: "RAG",

    blurb:
      "Framework-agnostic RAG auditing library that detects stale or conflicting context before LLM inference.",

    href: "https://github.com/ish4n10/staleguard",
  },

  {
    name: "SymExec",

    offset: "0x00A0",

    arch: "Python",

    tag: "Program Analysis",

    blurb:
      "Symbolic execution engine with SMT-backed feasibility checking and interactive control-flow visualization.",

    href: "https://github.com/ish4n10/wasm-sym",
  },
];

export const PROJECTS_HEADER = {
  index: "0x03",

  section: ".projects",

  symbol: "list_projects()",

  title: "Projects",

  subtitle:
    "Selected personal projects exploring databases, machine learning, and program analysis.",
};

export const WRITING_HEADER = {
  index: "0x04",

  section: ".text",

  symbol: "list_articles()",

  title: "Writing",

  subtitle:
    "Notes on machine learning, systems programming, and software engineering.",

  objdump: "$ objdump -d ./writing/",
};

export const CONTACT = {
  header: {
    index: "0x05",

    section: ".plt",

    symbol: "extern contact()",

    title: "Contact",

    subtitle:
      "Available for software engineering, backend, and AI infrastructure opportunities.",
  },

  heading: [
    "Get in touch.",
    "Let's build something."
  ],

  links: [
    {
      label: "ishan.tripathi010@gmail.com",
      href: "mailto:ishan.tripathi010@gmail.com",
      sym: "mail"
    },

    {
      label: "github.com/ish4n10",
      href: "https://github.com/ish4n10",
      sym: "github"
    },

    {
      label: "linkedin.com/in/ish4n10",
      href: "https://linkedin.com/in/ish4n10",
      sym: "linkedin"
    }
  ]
};

export const FOOTER = {
  left: "build/ishan.elf",
  center: "sha256: dead…beef",
  right: "Ishan Tripathi",
};
