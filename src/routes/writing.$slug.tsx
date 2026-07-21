import { useEffect, useRef, useState } from "react";
import { createFileRoute, Link, notFound } from "@tanstack/react-router";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import "highlight.js/styles/github.css";
import "katex/dist/katex.min.css";
import { ArrowUpRight, BookOpen, List, Moon, Sun } from "lucide-react";
import { getPost, POSTS, type Heading } from "@/lib/posts";

export const Route = createFileRoute("/writing/$slug")({
  loader: ({ params }) => {
    const post = getPost(params.slug);
    if (!post) throw notFound();
    return { post };
  },
  head: ({ loaderData }) => {
    if (!loaderData) {
      return {
        meta: [
          { title: "Not found — ishan.elf" },
          { name: "robots", content: "noindex" },
        ],
      };
    }
    const { post } = loaderData;
    return {
      meta: [
        { title: `${post.title} — ishan.elf` },
        { name: "description", content: post.summary },
        { property: "og:title", content: post.title },
        { property: "og:description", content: post.summary },
        { property: "og:type", content: "article" },
      ],
    };
  },
  component: PostPage,
  notFoundComponent: PostNotFound,
});

function PostNotFound() {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-3xl px-6 py-24">
        <p className="text-mono text-accent">// SIGSEGV · symbol not found</p>
        <h1 className="mt-4 font-serif text-4xl">No such symbol.</h1>
        <p className="mt-4 text-muted-foreground">
          The address you requested isn&rsquo;t mapped into this binary.
        </p>
        <Link
          to="/"
          hash="writing"
          className="text-mono mt-8 inline-flex items-center gap-1 text-accent hover:underline"
        >
          &larr; return to .text
        </Link>
      </div>
    </div>
  );
}

type MobilePanel = "none" | "index" | "toc";

function PostPage() {
  const { post } = Route.useLoaderData();
  const [mobilePanel, setMobilePanel] = useState<MobilePanel>("none");
  const darkRef = useRef(false);
  const [, forceRender] = useState(0);

  useEffect(() => {
    const stored = localStorage.getItem("theme");
    const isDark = stored === "dark";
    darkRef.current = isDark;
    document.documentElement.classList.toggle("dark", isDark);
    forceRender(n => n + 1);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {/* Header bar styled like a disassembler window */}
      <header className="sticky top-0 z-40 border-b border-border/60 bg-background/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-6 py-3">
          <Link to="/" className="font-mono text-sm">
            <span className="text-accent">$</span> ./ishan
            <span className="opacity-40">.elf</span>
          </Link>
          <div className="flex items-center gap-4">
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
            <Link
              to="/"
              hash="writing"
              data-cursor="ret"
              className="text-mono text-muted-foreground hover:text-foreground"
            >
              &larr; .text index
            </Link>
          </div>
        </div>
      </header>

      <div className="flex justify-center">
        {/* Desktop sidebar: symbol table index (fixed to left edge) */}
        <aside className="fixed left-0 top-14 hidden w-64 border-r border-border lg:block">
          <nav className="max-h-[calc(100vh-3.5rem)] overflow-y-auto px-4 py-8">
            <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              .symtab
            </div>
            <ul className="mt-4 space-y-1">
              {POSTS.map((p) => {
                const active = p.slug === post.slug;
                return (
                  <li key={p.slug}>
                    <Link
                      to="/writing/$slug"
                      params={{ slug: p.slug }}
                      data-cursor="objdump"
                      className={`group flex flex-col gap-0.5 rounded-md px-3 py-2 text-sm transition-colors ${
                        active
                          ? "bg-accent/10 text-accent"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <span className="font-mono text-[10px] opacity-50">
                        {p.date}
                      </span>
                      <span className="font-medium leading-tight">
                        {p.title}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
        </aside>

        <div className="flex w-full max-w-7xl">
          {/* Spacer matching fixed sidebar */}
          <div className="hidden w-64 shrink-0 lg:block" />

          {/* Main content */}
          <main className="min-w-0 flex-1 px-6 py-16">
          <div className="mx-auto max-w-3xl">
            {/* Symbol banner */}
            <div className="border border-border bg-card">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border bg-secondary/60 px-4 py-2 font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
                <span className="flex items-center gap-3">
                  <span className="text-accent">{post.offset}</span>
                  <span className="text-foreground/70">
                    &lt;{post.symbol}&gt;:
                  </span>
                </span>
                <span className="flex items-center gap-3">
                  <span>Section {post.section}</span>
                  <span className="opacity-60">|</span>
                  <span>{post.date}</span>
                </span>
              </div>
              <div className="grid grid-cols-1 gap-6 p-6 sm:grid-cols-12">
                <div className="sm:col-span-3">
                  <p className="text-mono text-accent">{post.kicker}</p>
                  <p className="text-mono mt-3 opacity-70">
                    {post.words.toLocaleString()} words
                  </p>
                  <p className="text-mono opacity-70">{post.read}</p>
                </div>
                <div className="sm:col-span-9">
                  <h1 className="font-serif text-4xl leading-tight md:text-5xl">
                    {post.title}
                  </h1>
                  <p className="mt-4 text-muted-foreground">{post.summary}</p>
                  <div className="mt-4 flex flex-wrap items-center gap-2">
                    {post.tags.map((t: string) => (
                      <span
                        key={t}
                        className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground before:mr-1 before:content-['#']"
                      >
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Markdown body */}
            <article className="prose-rev mt-12">
              <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkMath]}
                rehypePlugins={[rehypeHighlight, rehypeKatex, rehypeSlug]}
              >
                {post.content}
              </ReactMarkdown>
            </article>

            {/* Footer stub */}
            <div className="mt-16 border-t border-border pt-6">
              <p className="text-mono opacity-60">
                /* end of &lt;{post.symbol}&gt; &mdash; ret */
              </p>
              <div className="mt-6 flex flex-wrap items-center justify-between gap-4">
                <Link
                  to="/"
                  hash="writing"
                  className="text-mono inline-flex items-center gap-1 text-accent hover:underline"
                >
                  &larr; back to .text
                </Link>
                <div className="text-mono flex flex-wrap gap-4 opacity-70">
                  {POSTS.filter((p) => p.slug !== post.slug).map((p) => (
                      <Link
                        key={p.slug}
                        to="/writing/$slug"
                        params={{ slug: p.slug }}
                        data-cursor="objdump"
                        className="inline-flex items-center gap-1 hover:text-accent"
                      >
                        {p.title}
                        <ArrowUpRight className="h-3 w-3" />
                      </Link>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Desktop sidebar: table of contents */}
        <aside className="hidden w-64 shrink-0 border-l border-border lg:block">
          <nav className="sticky top-14 max-h-[calc(100vh-3.5rem)] overflow-y-auto px-4 py-8">
            <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              .headers
            </div>
            {post.headings.length === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground/50">(no symbols)</p>
            ) : (
              <ul className="mt-4 space-y-1">
                {post.headings.map((h: Heading) => (
                  <li key={h.id}>
                    <a
                      href={`#${h.id}`}
                      className={`block rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground ${
                        h.level === 3 ? "pl-7" : ""
                      }`}
                    >
                      {h.text}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </nav>
        </aside>
        </div>
      </div>

      {/* Mobile bottom tab bar */}
      <div className="fixed inset-x-0 bottom-0 z-50 border-t border-border bg-background/95 backdrop-blur lg:hidden">
        <div className="flex items-center justify-around py-2">
          <button
            onClick={() =>
              setMobilePanel(mobilePanel === "index" ? "none" : "index")
            }
            className={`flex flex-col items-center gap-0.5 px-4 py-1 text-[10px] uppercase tracking-wider transition-colors ${
              mobilePanel === "index"
                ? "text-accent"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <List className="h-4 w-4" />
            .symtab
          </button>
          <button
            onClick={() =>
              setMobilePanel(mobilePanel === "toc" ? "none" : "toc")
            }
            className={`flex flex-col items-center gap-0.5 px-4 py-1 text-[10px] uppercase tracking-wider transition-colors ${
              mobilePanel === "toc"
                ? "text-accent"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <BookOpen className="h-4 w-4" />
            .headers
          </button>
        </div>
      </div>

      {/* Mobile overlay backdrop */}
      <div
        className={`fixed inset-0 z-40 bg-black/40 transition-opacity duration-300 lg:hidden ${
          mobilePanel === "none"
            ? "pointer-events-none opacity-0"
            : "opacity-100"
        }`}
        onClick={() => setMobilePanel("none")}
      />

      {/* Mobile slide-up panel */}
      <div
        className={`fixed inset-x-0 bottom-[49px] z-40 max-h-[50vh] overflow-y-auto border-t border-border bg-background transition-transform duration-300 lg:hidden ${
          mobilePanel === "none" ? "translate-y-full" : "translate-y-0"
        }`}
      >
        {mobilePanel === "index" && (
          <div className="px-4 py-6">
            <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              .symtab
            </div>
            <ul className="mt-4 space-y-1">
              {POSTS.map((p) => {
                const active = p.slug === post.slug;
                return (
                  <li key={p.slug}>
                    <Link
                      to="/writing/$slug"
                      params={{ slug: p.slug }}
                      onClick={() => setMobilePanel("none")}
                      className={`group flex flex-col gap-0.5 rounded-md px-3 py-2 text-sm transition-colors ${
                        active
                          ? "bg-accent/10 text-accent"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                    >
                      <span className="font-mono text-[10px] opacity-50">
                        {p.date}
                      </span>
                      <span className="font-medium leading-tight">
                        {p.title}
                      </span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
        {mobilePanel === "toc" && (
          <div className="px-4 py-6">
            <div className="font-mono text-[11px] uppercase tracking-wider text-muted-foreground">
              .headers
            </div>
            {post.headings.length === 0 ? (
              <p className="mt-4 text-sm text-muted-foreground/50">
                (no symbols)
              </p>
            ) : (
              <ul className="mt-4 space-y-1">
                {post.headings.map((h: Heading) => (
                  <li key={h.id}>
                    <a
                      href={`#${h.id}`}
                      onClick={() => setMobilePanel("none")}
                      className={`block rounded-md px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground ${
                        h.level === 3 ? "pl-7" : ""
                      }`}
                    >
                      {h.text}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>

      {/* Bottom spacer so content isnt hidden behind the mobile bar */}
      <div className="h-[49px] lg:hidden" />
    </div>
  );
}
