import { Highlight } from "prism-react-renderer";
import { useEffect, useRef } from "react";
import { microGptTheme } from "./syntax";

type CodeViewerProps = {
  code: string;
  highlightLine?: number | null;
};

export function CodeViewer({ code, highlightLine }: CodeViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (highlightLine == null) return;
    const el = containerRef.current?.querySelector<HTMLDivElement>(
      `[data-line="${highlightLine}"]`
    );
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [highlightLine]);

  return (
    <div
      ref={containerRef}
      className="h-full overflow-auto text-[12.5px] leading-[1.55] font-mono"
    >
      <Highlight code={code} language="python" theme={microGptTheme}>
        {({ className, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={`${className} px-0 py-4 m-0 min-w-fit`}
            style={{ ...style, backgroundColor: "transparent" }}
          >
            {tokens.map((line, i) => {
              const lineNumber = i + 1;
              const isHighlighted = highlightLine === lineNumber;
              const { style: lineStyle, ...lineRest } = getLineProps({ line });
              return (
                <div
                  key={i}
                  {...lineRest}
                  data-line={lineNumber}
                  style={lineStyle}
                  className={`flex pr-4 transition-colors ${
                    isHighlighted ? "bg-violet-500/10" : "hover:bg-white/[0.025]"
                  }`}
                >
                  <span
                    className={`select-none w-10 pr-3 text-right shrink-0 ${
                      isHighlighted ? "text-violet-300" : "text-ink-400"
                    }`}
                  >
                    {lineNumber}
                  </span>
                  <span className="whitespace-pre">
                    {line.map((token, j) => {
                      const { style: tokenStyle, ...tokenRest } = getTokenProps({ token });
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
