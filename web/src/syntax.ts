import type { PrismTheme } from "prism-react-renderer";

// Shared Prism theme used by both the single-pane CodeViewer and the
// side-by-side CodeDiff. Kept in its own module so consumers can import it
// without forcing those component files to mix component and non-component
// exports (Vite's React Fast Refresh requires component-only modules).
export const microGptTheme: PrismTheme = {
  plain: { color: "#e6e8ef", backgroundColor: "transparent" },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#565b6e", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#8b91a6" } },
    { types: ["string", "char", "string-interpolation", "attr-value"], style: { color: "#67e8a4" } },
    { types: ["number", "boolean", "constant"], style: { color: "#7dd3fc" } },
    { types: ["keyword", "selector", "atrule"], style: { color: "#a78bfa" } },
    { types: ["builtin", "class-name", "maybe-class-name"], style: { color: "#fbbf24" } },
    { types: ["function", "decorator", "attr-name"], style: { color: "#f0abfc" } },
    { types: ["operator"], style: { color: "#c8cbd6" } },
    { types: ["variable", "tag"], style: { color: "#e6e8ef" } },
    { types: ["triple-quoted-string"], style: { color: "#67e8a4", fontStyle: "italic" } },
  ],
};
