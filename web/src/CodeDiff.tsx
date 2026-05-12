import { Highlight, type Token } from "prism-react-renderer";
import { diffLines } from "diff";
import { useMemo } from "react";
import { microGptTheme } from "./syntax";

// Side-by-side line-level diff for two Python sources. We deliberately keep
// this dumb for v1: walk `diffLines` hunks linearly and emit one paired row
// per diff line, with `null` on the side that has no content for that row.
// That keeps the left and right columns the SAME row count, so a plain
// 2-col grid renders them aligned without subgrid gymnastics. A future
// concept-aware variant will replace the row builder here without touching
// the renderer.

type LineEntry = { n: number; text: string };
type DiffRow = {
  left: LineEntry | null;
  right: LineEntry | null;
  kind: "same" | "del" | "add";
};

function buildRows(leftCode: string, rightCode: string): DiffRow[] {
  const changes = diffLines(leftCode, rightCode);
  const rows: DiffRow[] = [];
  let leftN = 0;
  let rightN = 0;
  for (const change of changes) {
    // `change.value` is the concatenated content of this hunk and almost
    // always ends with a trailing newline; splitting on "\n" then dropping
    // the trailing empty entry recovers the original line list. If the
    // file's last line had no newline, no empty entry exists and the pop
    // is a no-op — which is exactly what we want.
    const lines = change.value.split("\n");
    if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
    if (change.added) {
      for (const text of lines) {
        rightN += 1;
        rows.push({ kind: "add", left: null, right: { n: rightN, text } });
      }
    } else if (change.removed) {
      for (const text of lines) {
        leftN += 1;
        rows.push({ kind: "del", left: { n: leftN, text }, right: null });
      }
    } else {
      for (const text of lines) {
        leftN += 1;
        rightN += 1;
        rows.push({
          kind: "same",
          left: { n: leftN, text },
          right: { n: rightN, text },
        });
      }
    }
  }
  return rows;
}

function rowBg(kind: DiffRow["kind"], hasEntry: boolean): string {
  if (kind === "same") return "";
  if (hasEntry) {
    return kind === "del" ? "bg-rose-500/10" : "bg-emerald-500/10";
  }
  // Placeholder on the side that has no content for this diff row.
  return "bg-ink-800/30";
}

function DiffColumn({
  rows,
  side,
  code,
}: {
  rows: DiffRow[];
  side: "left" | "right";
  code: string;
}) {
  return (
    <div className="overflow-x-auto min-w-0">
      <Highlight code={code} language="python" theme={microGptTheme}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre
            className="m-0 py-4 min-w-fit"
            style={{ backgroundColor: "transparent" }}
          >
            {rows.map((row, i) => {
              const entry = row[side];
              const bg = rowBg(row.kind, entry !== null);
              if (!entry) {
                return (
                  <div key={i} className={`flex pr-4 ${bg}`}>
                    <span className="select-none w-10 pr-3 text-right shrink-0 text-ink-500">
                      ·
                    </span>
                    <span className="whitespace-pre">{"\u00A0"}</span>
                  </div>
                );
              }
              const line: Token[] = tokens[entry.n - 1] ?? [];
              const { style: lineStyle, ...lineRest } = getLineProps({ line });
              return (
                <div
                  key={i}
                  {...lineRest}
                  data-line={entry.n}
                  style={lineStyle}
                  className={`flex pr-4 ${bg}`}
                >
                  <span className="select-none w-10 pr-3 text-right shrink-0 text-ink-400">
                    {entry.n}
                  </span>
                  <span className="whitespace-pre">
                    {line.map((token, j) => {
                      const { style: tokenStyle, ...tokenRest } = getTokenProps({
                        token,
                      });
                      return <span key={j} {...tokenRest} style={tokenStyle} />;
                    })}
                    {line.length === 0 ? "\u00A0" : null}
                  </span>
                </div>
              );
            })}
          </pre>
        )}
      </Highlight>
    </div>
  );
}

type CodeDiffProps = {
  leftCode: string;
  rightCode: string;
};

export function CodeDiff({ leftCode, rightCode }: CodeDiffProps) {
  const rows = useMemo(() => buildRows(leftCode, rightCode), [leftCode, rightCode]);
  return (
    <div className="h-full overflow-y-auto text-[12.5px] leading-[1.55] font-mono">
      <div className="grid grid-cols-2 divide-x divide-ink-800/60 min-h-full">
        <DiffColumn rows={rows} side="left" code={leftCode} />
        <DiffColumn rows={rows} side="right" code={rightCode} />
      </div>
    </div>
  );
}
