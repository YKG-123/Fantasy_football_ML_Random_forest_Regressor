import "./index.css";
import data from "../../dump/test.json";

import { Table } from "./Table";
import logo from "./logo.svg";
import reactLogo from "./react.svg";

export function App() {
  return (
    <div className="app">
      <h1>Bun + React</h1>
      <p>
        Edit <code>src/App.tsx</code> and save to test HMR
      </p>
      <Table json={data} />
    </div>
  );
}

export default App;
