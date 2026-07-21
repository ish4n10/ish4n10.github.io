interface Frontmatter {
  title?: string;
  tags?: string[];
  [key: string]: unknown;
}

function parseFrontmatter(raw: string): { data: Frontmatter; content: string } {
  raw = raw.replace(/^\uFEFF/, "");
  const match = raw.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
  if (!match) return { data: {}, content: raw };
  const yaml = match[1];
  const data: Frontmatter = {};
  let currentKey = "";
  for (const line of yaml.split("\n")) {
    const listItem = line.match(/^\s+-\s+(.+)/);
    if (listItem && currentKey) {
      const arr = (data[currentKey] as string[]) || [];
      arr.push(listItem[1].replace(/^["']|["']$/g, ""));
      data[currentKey] = arr;
      continue;
    }
    const kv = line.match(/^(\w+):\s*(.*)/);
    if (kv) {
      currentKey = kv[1];
      const val = kv[2].trim().replace(/^["']|["']$/g, "");
      data[currentKey] = val || ([] as string[]);
    }
  }
  return { data, content: raw.slice(match[0].length) };
}

export interface Heading {
  level: number;
  text: string;
  id: string;
}

export interface Post {
  slug: string;
  offset: string;
  date: string;
  section: string;
  symbol: string;
  title: string;
  kicker: string;
  words: number;
  read: string;
  tags: string[];
  summary: string;
  content: string;
  headings: Heading[];
}

const modules = import.meta.glob("../content/posts/*.md", {
  query: "?raw",
  import: "default",
  eager: true,
}) as Record<string, string>;

const BASE_OFFSET = 0x00401A20;

function wordCount(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

function readTime(words: number): string {
  return `${Math.max(1, Math.round(words / 220))} min`;
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function extractHeadings(content: string): Heading[] {
  const headingRe = /^(#{2,3})\s+(.+)$/gm;
  const headings: Heading[] = [];
  let m: RegExpExecArray | null;
  while ((m = headingRe.exec(content)) !== null) {
    const text = m[2].replace(/[*_`]/g, "").trim();
    if (text) headings.push({ level: m[1].length, text, id: slugify(text) });
  }
  return headings;
}

function extractSummary(content: string): string {
  const clean = content.replace(/^#+\s+.*$/m, "").trim();
  const first = clean.split("\n\n").find((p) => p.trim().length > 20);
  if (!first) return "";
  return first.replace(/[*_`#\[\]]/g, "").slice(0, 200).trim();
}

function slugToSymbol(slug: string): string {
  const parts = slug.split("-");
  if (parts.length <= 2) return `${slug}::entry`;
  return `${parts.slice(0, -2).join("_")}::${parts.slice(-2).join("_")}`;
}

function postFromModule(path: string, raw: string, index: number): Post {
  const filename = path.split("/").pop()!;
  const datePrefix = filename.slice(0, 10);
  const slug = filename.replace(/^\d{4}-\d{2}-\d{2}-/, "").replace(/\.md$/, "");
  const { data, content } = parseFrontmatter(raw);
  const wc = wordCount(content);
  const tags = data.tags || [];
  const kicker = tags.length > 0 ? tags[0] : "writing";
  const yyyy = datePrefix.slice(0, 4);
  const monthNum = parseInt(datePrefix.slice(5, 7), 10);
  const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const monthName = monthNames[monthNum - 1] || datePrefix.slice(5, 7);

  return {
    slug,
    offset: `0x${(BASE_OFFSET + index * 0x20).toString(16).toUpperCase().padStart(8, "0")}`,
    date: `${yyyy} · ${monthName}`,
    section: ".text",
    symbol: slugToSymbol(slug),
    title: data.title || slug,
    kicker,
    words: wc,
    read: readTime(wc),
    tags,
    summary: extractSummary(content),
    headings: extractHeadings(content),
    content,
  };
}

const POSTS: Post[] = Object.entries(modules)
  .map(([path, raw], index) => postFromModule(path, raw as string, index))
  .sort((a, b) => {
    const dateA = a.date.split(" · ")[0];
    const dateB = b.date.split(" · ")[0];
    return dateB.localeCompare(dateA);
  });

export function getPost(slug: string): Post | undefined {
  return POSTS.find((p) => p.slug === slug);
}

export { POSTS };
