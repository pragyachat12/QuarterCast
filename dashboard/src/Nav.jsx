import { NavLink } from 'react-router-dom'

export default function Nav() {
  return (
    <header className="app-header">
      <div className="header-left">
        <span className="logo">QuarterCast</span>
        <span className="header-sub">Live earnings-surprise predictions, explained</span>
      </div>
      <nav className="top-nav">
        <NavLink to="/" end className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>About</NavLink>
        <NavLink to="/dashboard" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Dashboard</NavLink>
        <NavLink to="/analysis" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>Analysis</NavLink>
      </nav>
    </header>
  )
}
