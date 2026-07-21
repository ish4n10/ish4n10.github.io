import fs from "fs";
import path from "path";

const outDir = path.resolve(process.argv[2] || ".output/public");

const assetsDir = path.join(outDir, "assets");
if (!fs.existsSync(assetsDir)) {
  console.error("Assets directory not found:", assetsDir);
  process.exit(1);
}

const files = fs.readdirSync(assetsDir);
const mainJs = files.find((f) => /^index-[A-Za-z0-9]+\.js$/.test(f));
const mainCss = files.find((f) => /^index-[A-Za-z0-9]+\.css$/.test(f));
const stylesCss = files.find((f) => /^styles-[A-Za-z0-9]+\.css$/.test(f));

const cssLinks = [mainCss, stylesCss].filter(Boolean).map((f) => `<link rel="stylesheet" href="/assets/${f}" />`).join("\n    ");

const html = `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Ishaan — Deeptech, AI & Low-level Security</title>
    <meta name="description" content="Personal site of Ishaan (ish4n10). Notes and work on deeptech, AI systems, and low-level security." />
    ${cssLinks}
  </head>
  <body>
    <div id="root"></div>
    ${mainJs ? `<script type="module" crossorigin src="/assets/${mainJs}"></script>` : ""}
  </body>
</html>`;

fs.writeFileSync(path.join(outDir, "index.html"), html);
console.log("Generated index.html in", outDir);
