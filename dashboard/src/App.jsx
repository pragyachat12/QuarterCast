import { Routes, Route } from 'react-router-dom'
import Nav from './Nav'
import About from './pages/About'
import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <Nav />
      <Routes>
        <Route path="/" element={<About />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/analysis" element={<Analysis />} />
      </Routes>
    </div>
  )
}
