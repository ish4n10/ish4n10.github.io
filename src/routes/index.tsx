import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowUpRight, Github, Linkedin, Mail, Menu, Moon, Sun, X } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { POSTS } from "@/lib/posts";
import {
  NAV,
  HERO,
  ABOUT,
  EXPERIENCE,
  WORK_HEADER,
  PROJECTS,
  PROJECTS_HEADER,
  WRITING_HEADER,
  CONTACT,
  FOOTER,
} from "@/content/site";

export const Route = createFileRoute("/")({
  component: Index,
});


function SectionHeader({
  index,
  section,
  symbol,
  title,
  subtitle,
}: {
  index: string;
  section: string;
  symbol: string;
  title: string;
  subtitle?: ReactNode;
}) {
  return (
    <div>
      <p className="text-mono">
        <span className="text-accent">{index}</span>
        <span className="mx-2 opacity-40">·</span>
        <span>SECTION {section}</span>
      </p>
      <h2 className="mt-3 font-serif text-3xl leading-tight md:text-4xl">
        {title}
      </h2>
      <p className="text-mono mt-3 opacity-70">
        &lt;{symbol}&gt;
      </p>
      {subtitle ? (
        <p className="mt-4 max-w-xs text-sm leading-relaxed text-muted-foreground">
          {subtitle}
        </p>
      ) : null}
    </div>
  );
}

function Index() {
  const darkRef = useRef(false);
  const [, forceRender] = useState(0);
  const [mobileMenu, setMobileMenu] = useState(false);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const isDark = stored === "dark";
    darkRef.current = isDark;
    document.documentElement.classList.toggle("dark", isDark);
    forceRender(n => n + 1);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Nav — styled like a symbol table */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-5xl items-center justify-between gap-4 px-6 py-3">
          <a
            href="#top"
            className="font-mono text-sm tracking-wide text-foreground"
          >
            <span className="text-accent">$</span> ./ishan
            <span className="opacity-40">.elf</span>
          </a>
          <nav className="hidden gap-6 sm:flex">
            {NAV.map((n) => (
              <a
                key={n.href}
                href={n.href}
                data-cursor={
                  n.label === "about" ? ".about" :
                  n.label === "work" ? "trace()" :
                  n.label === "writing" ? "objdump" :
                  n.label === "contact" ? ".contact" : undefined
                }
                className="text-mono flex items-baseline gap-1 text-muted-foreground transition-colors hover:text-foreground"
              >
                <span className="opacity-50">{n.addr}</span>
                <span>{n.label}</span>
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-2 sm:gap-3">
            <button
              type="button"
              onClick={() => setMobileMenu(true)}
              className="p-2 text-foreground hover:text-accent transition-colors sm:hidden"
              aria-label="Open menu"
            >
              <Menu className="h-5 w-5" />
            </button>
            <button
              type="button"
              data-cursor="call"
              onClick={() => {
                const next = !darkRef.current;
                darkRef.current = next;
                document.documentElement.classList.toggle("dark", next);
                localStorage.setItem("theme", next ? "dark" : "light");
                forceRender(n => n + 1);
              }}
              className="p-2 text-foreground hover:text-accent transition-colors"
              aria-label="Toggle theme"
            >
              {darkRef.current ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
            <a
              href="https://github.com/ish4n10"
              target="_blank"
              rel="noreferrer"
              data-cursor="git push"
              className="text-mono inline-flex items-center gap-1 text-foreground hover:text-accent"
            >
              ish4n10 <ArrowUpRight className="h-3 w-3" />
            </a>
          </div>
        </div>
      </header>

      {/* Mobile side panel */}
      <div
        className={`fixed inset-0 z-50 transition-opacity duration-300 sm:hidden ${
          mobileMenu ? "opacity-100" : "pointer-events-none opacity-0"
        }`}
        onClick={() => setMobileMenu(false)}
      >
        <div className="absolute inset-0 bg-black/40" />
      </div>
      <div
        className={`fixed right-0 top-0 z-50 h-full w-64 border-l border-border bg-background transition-transform duration-300 sm:hidden ${
          mobileMenu ? "translate-x-0" : "translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <span className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
            .symtab
          </span>
          <button
            type="button"
            onClick={() => setMobileMenu(false)}
            className="p-1 text-foreground hover:text-accent transition-colors"
            aria-label="Close menu"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <nav className="px-4 py-6">
          <ul className="space-y-1">
            {NAV.map((n) => (
              <li key={n.href}>
                <a
                  href={n.href}
                  onClick={() => setMobileMenu(false)}
                  data-cursor={
                    n.label === "about" ? ".about" :
                    n.label === "work" ? "trace()" :
                    n.label === "writing" ? "objdump" :
                    n.label === "contact" ? ".contact" : undefined
                  }
                  className="text-mono flex items-baseline gap-3 rounded-md px-3 py-2 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                >
                  <span className="opacity-50">{n.addr}</span>
                  <span>{n.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>

      <main id="top" className="mx-auto max-w-5xl px-6">
        {/* Hero — framed as an ELF-style header block */}
        <section className="grid grid-cols-1 gap-10 border-b border-border py-20 md:grid-cols-12 md:py-28">
          <div className="md:col-span-8">
            <p className="text-mono mb-6 text-accent">
              {HERO.header}
            </p>
            <h1 className="font-serif text-5xl leading-[1.02] tracking-tight md:text-7xl">
              {HERO.title[0]}
              <br />
              <span className="italic text-accent">{HERO.title[1]}</span> {HERO.title[2]}
            </h1>
            <p className="mt-8 max-w-xl text-lg leading-relaxed text-muted-foreground">
              {HERO.subtitle}
            </p>

            <dl className="text-mono mt-10 grid max-w-md grid-cols-[auto_1fr] gap-x-6 gap-y-1 border-l-2 border-accent/60 pl-4">
              {HERO.readelf.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="opacity-60">{k}:</dt>
                  <dd className="text-foreground">{v}</dd>
                </div>
              ))}
            </dl>
          </div>

          <aside className="md:col-span-4 md:border-l md:border-border md:pl-8">
            <p className="text-mono mb-4">
              <span className="text-accent">$</span> ps -o current
            </p>
            <ul className="space-y-4 text-sm">
              {HERO.ps.map(([k, v]) => (
                <li key={k}>
                  <span className="text-mono opacity-70">{k} =</span>
                  <br />
                  <span className="font-serif text-base">{v}</span>
                </li>
              ))}
            </ul>
          </aside>
        </section>

        {/* About */}
        <section
          id="about"
          className="grid grid-cols-1 gap-10 border-b border-border py-20 md:grid-cols-12"
        >
          <div className="md:col-span-4">
            <SectionHeader
              index={ABOUT.index}
              section={ABOUT.section}
              symbol={ABOUT.symbol}
              title={ABOUT.title}
            />
          </div>
          <div className="md:col-span-8">
            <div className="space-y-5 text-base leading-relaxed text-muted-foreground">
              {ABOUT.paragraphs.map((p, i) => (
                <p key={i}>{p}</p>
              ))}
            </div>

            <div className="mt-10 border-t border-border pt-4">
              <p className="text-mono mb-3 opacity-60">
                readelf &minus;S ./ishan &mdash; segments
              </p>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-4 sm:grid-cols-4">
                {ABOUT.segments.map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-mono text-accent">{k}</dt>
                    <dd className="mt-1 font-serif text-lg">{v}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </section>

        {/* Experience — call graph style */}
        <section
          id="work"
          className="grid grid-cols-1 gap-10 border-b border-border py-20 md:grid-cols-12"
        >
          <div className="md:col-span-4">
            <SectionHeader
              index={WORK_HEADER.index}
              section={WORK_HEADER.section}
              symbol={WORK_HEADER.symbol}
              title={WORK_HEADER.title}
              subtitle={WORK_HEADER.subtitle}
            />
          </div>
          <ol className="md:col-span-8 relative space-y-8 border-l border-border pl-6">
            {EXPERIENCE.map((e) => (
              <li key={e.role} className="relative">
                {/* node */}
                <span className="absolute -left-[29px] top-2 h-2.5 w-2.5 rotate-45 border border-accent bg-background" />
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="text-mono">
                    <span className="text-accent">{e.offset}</span>
                    <span className="mx-2 opacity-40">·</span>
                    <span>&lt;{e.symbol}&gt;</span>
                  </p>
                  <p className="text-mono opacity-70">{e.period}</p>
                </div>
                <h3 className="mt-2 font-serif text-2xl">
                  {e.role}{" "}
                  <span className="text-muted-foreground">
                    &mdash; {e.org}
                  </span>
                </h3>
                <p className="mt-2 max-w-2xl leading-relaxed text-muted-foreground">
                  {e.notes}
                </p>
                <div className="mt-3 flex flex-wrap gap-x-3 gap-y-1">
                  {e.stack.map((s) => (
                    <span key={s} className="text-mono opacity-70">
                      [{s}]
                    </span>
                  ))}
                </div>
              </li>
            ))}
          </ol>
        </section>

        {/* Projects — binary listing */}
        <section
          id="projects"
          className="grid grid-cols-1 gap-10 border-b border-border py-20 md:grid-cols-12"
        >
          <div className="md:col-span-4">
            <SectionHeader
              index={PROJECTS_HEADER.index}
              section={PROJECTS_HEADER.section}
              symbol={PROJECTS_HEADER.symbol}
              title={PROJECTS_HEADER.title}
              subtitle={PROJECTS_HEADER.subtitle}
            />
          </div>
          <div className="md:col-span-8">
            {/* header row */}
            <div className="text-mono grid grid-cols-12 gap-3 border-b border-border pb-2 opacity-60">
              <span className="col-span-2">size</span>
              <span className="col-span-4">name</span>
              <span className="col-span-6">arch</span>
            </div>
            <ul className="divide-y divide-border">
              {PROJECTS.map((p) => (
                <li key={p.name}>
                  <a
                    href={p.href}
                    target="_blank"
                    rel="noreferrer"
                    data-cursor={p.name}
                    className="group block py-6"
                  >
                    <div className="grid grid-cols-12 items-baseline gap-3">
                      <span className="text-mono col-span-2 text-accent">
                        {p.offset}
                      </span>
                      <h3 className="col-span-4 font-serif text-2xl transition-colors group-hover:text-accent">
                        {p.name}
                      </h3>
                      <div className="col-span-5 flex items-baseline gap-2">
                        <span className="text-mono opacity-70">{p.arch}</span>
                      </div>
                      <div className="col-span-1 text-right">
                        <ArrowUpRight className="ml-auto h-5 w-5 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-accent" />
                      </div>
                    </div>
                    <div className="mt-3 grid grid-cols-12 gap-3">
                      <span className="text-mono col-span-2 opacity-60">
                        // {p.tag}
                      </span>
                      <p className="col-span-10 max-w-2xl leading-relaxed text-muted-foreground">
                        {p.blurb}
                      </p>
                    </div>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </section>

        {/* Writing */}
        <section
          id="writing"
          className="grid grid-cols-1 gap-10 border-b border-border py-20 md:grid-cols-12"
        >
          <div className="md:col-span-4">
            <SectionHeader
              index={WRITING_HEADER.index}
              section={WRITING_HEADER.section}
              symbol={WRITING_HEADER.symbol}
              title={WRITING_HEADER.title}
              subtitle={WRITING_HEADER.subtitle}
            />
            <p className="text-mono mt-6 opacity-60">
              {WRITING_HEADER.objdump}
            </p>
          </div>

          <div className="md:col-span-8 space-y-6">
            {POSTS.map((p, i) => (
              <Link
                key={p.slug}
                to="/writing/$slug"
                params={{ slug: p.slug }}
                data-cursor="objdump"
                className="group block border border-border bg-card transition-colors hover:border-accent"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border bg-secondary/60 px-4 py-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                  <span className="flex items-center gap-3">
                    <span className="text-accent">
                      {String(i).padStart(2, "0")}
                    </span>
                    <span>{p.offset}</span>
                    <span className="hidden text-foreground/70 sm:inline">
                      &lt;{p.symbol}&gt;:
                    </span>
                  </span>
                  <span className="flex items-center gap-3">
                    <span>Section {p.section}</span>
                    <span className="opacity-60">|</span>
                    <span>{p.date}</span>
                  </span>
                </div>

                <div className="grid grid-cols-1 gap-6 p-6 sm:grid-cols-12">
                  <div className="sm:col-span-3">
                    <p className="text-mono text-accent">{p.kicker}</p>
                    <p className="text-mono mt-3 opacity-70">
                      {p.words.toLocaleString()} words
                    </p>
                    <p className="text-mono opacity-70">{p.read}</p>
                  </div>
                  <div className="sm:col-span-9">
                    <h3 className="font-serif text-2xl leading-snug transition-colors group-hover:text-accent md:text-3xl">
                      {p.title}
                    </h3>

                    <pre className="mt-4 overflow-x-auto rounded-sm border border-border/60 bg-background/60 px-3 py-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
                      <span className="text-accent">{p.offset}</span>{"  "}
                      push   %rbp{"\n"}
                      <span className="text-accent">
                        {p.offset.replace(/.$/, "1")}
                      </span>
                      {"  "}mov    %rsp,%rbp{"\n"}
                      <span className="text-accent">
                        {p.offset.replace(/.$/, "4")}
                      </span>
                      {"  "}call   &lt;{p.symbol.split("::").pop()}&gt;
                    </pre>

                    <div className="mt-4 flex flex-wrap items-center gap-2">
                      {p.tags.map((t) => (
                        <span
                          key={t}
                          className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground before:mr-1 before:content-['#']"
                        >
                          {t}
                        </span>
                      ))}
                      <span className="ml-auto inline-flex items-center gap-1 font-mono text-[11px] uppercase tracking-wider text-foreground/80 transition-colors group-hover:text-accent">
                        Read
                        <ArrowUpRight className="h-3.5 w-3.5 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
                      </span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}

            <p className="text-mono opacity-60">
              /* end of .text &mdash; 2 symbols, more to come */
            </p>
          </div>
        </section>

        {/* Contact */}
        <section id="contact" className="grid grid-cols-1 gap-10 py-24 md:grid-cols-12">
          <div className="md:col-span-4">
            <SectionHeader
              index={CONTACT.header.index}
              section={CONTACT.header.section}
              symbol={CONTACT.header.symbol}
              title={CONTACT.header.title}
              subtitle={CONTACT.header.subtitle}
            />
          </div>
          <div className="md:col-span-8">
            <h2 className="font-serif text-4xl leading-[1.05] tracking-tight md:text-5xl">
              {CONTACT.heading[0]}{" "}
              <span className="italic text-accent">{CONTACT.heading[1]}</span>.
            </h2>

            <ul className="text-mono mt-10 divide-y divide-border border-y border-border">
              {(() => {
                const iconMap: Record<string, typeof Mail> = { Mail, Github, Linkedin };
                return CONTACT.links.map(({ label, href, sym }, i) => {
                  const iconKey = ["Mail", "Github", "Linkedin"][i] || "Mail";
                  const Icon = iconMap[iconKey] || Mail;
                  return (
                    <li key={label}>
                      <a
                        href={href}
                        target={href.startsWith("http") ? "_blank" : undefined}
                        rel="noreferrer"
                        className="group flex flex-wrap items-center gap-4 py-4 hover:text-accent"
                      >
                        <Icon className="h-4 w-4 opacity-70 group-hover:opacity-100" />
                        <span className="opacity-60">U</span>
                        <span>{sym}</span>
                        <span className="opacity-40">&rarr;</span>
                        <span className="text-foreground group-hover:text-accent">
                          {label}
                        </span>
                        <ArrowUpRight className="ml-auto h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" />
                      </a>
                    </li>
                  );
                });
              })()}
            </ul>
          </div>
        </section>
      </main>

      <footer className="border-t border-border">
        <div className="mx-auto grid max-w-5xl grid-cols-1 gap-3 px-6 py-8 text-sm text-muted-foreground sm:grid-cols-3 sm:items-center">
          <p className="text-mono opacity-70">
            {FOOTER.left.replace("$YEAR", String(new Date().getFullYear()))}
          </p>
          <p className="text-mono text-center opacity-70">
            {FOOTER.center}
          </p>
          <p className="text-mono sm:text-right">
            <span className="text-accent">$</span> {FOOTER.right}
          </p>
        </div>
      </footer>
    </div>
  );
}
