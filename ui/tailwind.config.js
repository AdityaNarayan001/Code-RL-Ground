/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'code-bg': '#1e1e1e',
        'code-comment': '#6a9955',
        'code-string': '#ce9178',
        'code-keyword': '#569cd6',
        'code-function': '#dcdcaa',
      },
    },
  },
  plugins: [],
}
