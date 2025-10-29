/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f5f7ff',
          100: '#ebefff',
          200: '#d6deff',
          300: '#b8c5ff',
          400: '#94a3ff',
          500: '#667eea',
          600: '#5568d3',
          700: '#4552b8',
          800: '#3a4494',
          900: '#2f3876',
        },
        secondary: {
          500: '#764ba2',
          600: '#6a4391',
          700: '#5d3b80',
        },
        dark: {
          50: '#e0e6ed',
          100: '#b3c0d1',
          200: '#8099b3',
          300: '#4d7195',
          400: '#26557f',
          500: '#1a1f3a',
          600: '#151a30',
          700: '#101426',
          800: '#0a0e1c',
          900: '#050712',
        }
      },
      backgroundImage: {
        'gradient-primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'gradient-dark': 'linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%)',
      },
      animation: {
        'slide-in': 'slideIn 0.3s ease-out',
        'fade-in': 'fadeIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        slideIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}