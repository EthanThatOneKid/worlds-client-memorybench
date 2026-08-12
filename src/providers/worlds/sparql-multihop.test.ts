import { describe, expect, it } from "bun:test"
import { Parser, Store, DataFactory } from "n3"
import { QueryEngine } from "@comunica/query-sparql-rdfjs-lite"
import { claimsToTurtle, type ExtractedClaim } from "./extraction"
import { SPARQL_PREFIXES, SCHEMA, RDF } from "./ontology"

const { namedNode } = DataFactory

function parseTurtleStore(turtle: string): Store {
  const parser = new Parser()
  const store = new Store()
  const quads = parser.parse(turtle)
  store.addQuads(quads)
  return store
}

describe("Multi-Hop SPARQL Domain Graph Reasoning", () => {
  const engine = new QueryEngine()

  it("queries multi-hop Person to Organization employment (schema:worksFor)", async () => {
    const claims: ExtractedClaim[] = [
      {
        domainClass: "Organization",
        subject: "Alice Smith",
        action: "works for",
        object: "Acme Corp",
        claimText: "Alice Smith works for Acme Corp as a software architect.",
      },
    ]

    const turtle = claimsToTurtle(claims, "session-test-01")
    const store = parseTurtleStore(turtle)

    // Direct N3 graph pattern check
    const personQuads = store.getQuads(null, namedNode(RDF.type), namedNode(SCHEMA.Person), null)
    expect(personQuads).toHaveLength(1)

    const personUri = personQuads[0]!.subject
    const worksForQuads = store.getQuads(personUri, namedNode(SCHEMA.worksFor), null, null)
    expect(worksForQuads).toHaveLength(1)

    const orgUri = worksForQuads[0]!.object
    const orgNameQuads = store.getQuads(orgUri, namedNode(SCHEMA.name), null, null)
    expect(orgNameQuads).toHaveLength(1)
    expect(orgNameQuads[0]!.object.value).toBe("Acme Corp")

    // Comunica SPARQL multi-hop SELECT query
    const query = `
      ${SPARQL_PREFIXES}
      SELECT ?personName ?orgName WHERE {
        ?person a schema:Person ;
                schema:name ?personName ;
                schema:worksFor ?org .
        ?org a schema:Organization ;
             schema:name ?orgName .
      }
    `

    const bindingsStream = await engine.queryBindings(query, { sources: [store] })
    const bindings = await bindingsStream.toArray()

    expect(bindings).toHaveLength(1)
    expect(bindings[0]!.get("personName")?.value).toBe("Alice Smith")
    expect(bindings[0]!.get("orgName")?.value).toBe("Acme Corp")
  })

  it("queries multi-hop Event about Person with status and provenance (schema:about, schema:eventStatus)", async () => {
    const claims: ExtractedClaim[] = [
      {
        domainClass: "Event",
        subject: "Bob Jones",
        action: "applied for",
        object: "visa extension",
        claimText: "Bob Jones applied for a visa extension in London.",
        when: "2023-06-10",
        where: "London",
        status: "Postponed",
      },
    ]

    const turtle = claimsToTurtle(claims, "session-test-02")
    const store = parseTurtleStore(turtle)

    const query = `
      ${SPARQL_PREFIXES}
      SELECT ?personName ?eventName ?status ?location WHERE {
        ?event a schema:Event ;
               schema:name ?eventName ;
               schema:about ?person ;
               schema:eventStatus ?status .
        ?person a schema:Person ;
                schema:name ?personName .
        OPTIONAL { ?event schema:location ?location }
      }
    `

    const bindingsStream = await engine.queryBindings(query, { sources: [store] })
    const bindings = await bindingsStream.toArray()

    expect(bindings).toHaveLength(1)
    expect(bindings[0]!.get("personName")?.value).toBe("Bob Jones")
    expect(bindings[0]!.get("eventName")?.value).toBe(
      "Bob Jones applied for a visa extension in London."
    )
    expect(bindings[0]!.get("status")?.value).toContain("EventPostponed")
    expect(bindings[0]!.get("location")?.value).toBe("London")
  })

  it("queries multi-hop Action with agent entity (schema:agent)", async () => {
    const claims: ExtractedClaim[] = [
      {
        domainClass: "Action",
        subject: "Charlie Brown",
        action: "relocated to",
        object: "San Francisco",
        claimText: "Charlie Brown relocated to San Francisco for a new role.",
      },
    ]

    const turtle = claimsToTurtle(claims, "session-test-03")
    const store = parseTurtleStore(turtle)

    const query = `
      ${SPARQL_PREFIXES}
      SELECT ?personName ?actionName ?target WHERE {
        ?act a schema:Action ;
             schema:name ?actionName ;
             schema:agent ?person .
        ?person a schema:Person ;
                schema:name ?personName .
        OPTIONAL { ?act schema:object ?target }
      }
    `

    const bindingsStream = await engine.queryBindings(query, { sources: [store] })
    const bindings = await bindingsStream.toArray()

    expect(bindings).toHaveLength(1)
    expect(bindings[0]!.get("personName")?.value).toBe("Charlie Brown")
    expect(bindings[0]!.get("target")?.value).toBe("San Francisco")
  })

  it("queries multi-hop MedicalCondition about Person (schema:MedicalCondition)", async () => {
    const claims: ExtractedClaim[] = [
      {
        domainClass: "MedicalCondition",
        subject: "Diana Prince",
        action: "diagnosed with",
        object: "Migraine",
        claimText: "Diana Prince was diagnosed with migraine headaches.",
      },
    ]

    const turtle = claimsToTurtle(claims, "session-test-04")
    const store = parseTurtleStore(turtle)

    const query = `
      ${SPARQL_PREFIXES}
      SELECT ?personName ?conditionName WHERE {
        ?med a schema:MedicalCondition ;
             schema:name ?conditionName ;
             schema:about ?person .
        ?person a schema:Person ;
                schema:name ?personName .
      }
    `

    const bindingsStream = await engine.queryBindings(query, { sources: [store] })
    const bindings = await bindingsStream.toArray()

    expect(bindings).toHaveLength(1)
    expect(bindings[0]!.get("personName")?.value).toBe("Diana Prince")
    expect(bindings[0]!.get("conditionName")?.value).toBe("Migraine")
  })
})
