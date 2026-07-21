import fs from "fs";
import path from "path";

const outDir = path.resolve(process.argv[2] || ".output/public");
const indexSrc = path.join(outDir, "index.html");
const dest = path.join(outDir, "404.html");

if (fs.existsSync(indexSrc)) {
  fs.copyFileSync(indexSrc, dest);
  console.log("Copied index.html → 404.html in", outDir);
} else {
  console.log("index.html not found in", outDir, "- skipping 404.html");
}

fs.writeFileSync(path.join(outDir, ".nojekyll"), "");
console.log("Created .nojekyll in", outDir);
