# gof/metamodel.py
#
# GoF JSON model — validation and comprehension metric.

VALID_CLASS_TYPES = {"class", "abstract", "interface"}

_CLASS_TYPE_ALIASES = {
    "abstract class": "abstract",
    "abstract_class": "abstract",
    "abstractclass": "abstract",
    "abstract interface": "interface",
    "interface class": "interface",
}

_REL_TYPE_ALIASES = {
    "extends": "inheritance",
    "implements": "implementation",
    "realizes": "implementation",
    "uses": "dependency",
    "has": "association",
    "contains": "composition",
    "has-a": "aggregation",
}
VALID_REL_TYPES = {
    "inheritance", "implementation",
    "association", "dependency",
    "composition", "aggregation",
}


def _validate_classes(classes: list, errors: list) -> set[str]:
    ids: set[str] = set()
    for cls in classes:
        cid = cls.get("id")
        if not cid:
            errors.append(f"Class missing 'id': {cls}")
            continue
        if cid in ids:
            errors.append(f"Duplicate id: {cid}")
        ids.add(cid)
        if cls.get("type") not in VALID_CLASS_TYPES:
            errors.append(f"Invalid class type '{cls.get('type')}' for {cid}")
    return ids


def _validate_relationships(rels: list, ids: set[str], errors: list) -> None:
    for rel in rels:
        if rel.get("type") not in VALID_REL_TYPES:
            errors.append(f"Invalid relationship type '{rel.get('type')}'")


def _normalise_identifiers(model: dict) -> None:
    for cls in model["classes"]:
        if cls.get("id"):
            cls["id"] = cls["id"].replace(" ", "")
        if cls.get("name"):
            cls["name"] = cls["name"].replace(" ", "")
    for rel in model["relationships"]:
        if rel.get("sourceId"):
            rel["sourceId"] = rel["sourceId"].replace(" ", "")
        if rel.get("targetId"):
            rel["targetId"] = rel["targetId"].replace(" ", "")


def _normalise_types(model: dict) -> None:
    for cls in model["classes"]:
        raw = (cls.get("type") or "").strip().lower()
        if raw not in VALID_CLASS_TYPES:
            cls["type"] = _CLASS_TYPE_ALIASES.get(raw, raw)
    for rel in model["relationships"]:
        raw = (rel.get("type") or "").strip().lower()
        if raw not in VALID_REL_TYPES:
            rel["type"] = _REL_TYPE_ALIASES.get(raw, raw)


def _drop_invalid_relationships(model: dict, ids: set[str]) -> None:
    model["relationships"] = [
        r for r in model["relationships"]
        if r.get("type") and r.get("sourceId") and r.get("targetId")
        and r["sourceId"] in ids and r["targetId"] in ids
    ]


def validate_gof(model: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(model.get("classes"), list):
        errors.append("Missing or invalid 'classes' array")
        return errors
    if not isinstance(model.get("relationships"), list):
        errors.append("Missing or invalid 'relationships' array")
        return errors
    _normalise_identifiers(model)
    _normalise_types(model)
    ids = _validate_classes(model["classes"], errors)
    _drop_invalid_relationships(model, ids)
    _validate_relationships(model["relationships"], ids, errors)
    return errors


def compute_comprehension(model: dict) -> float:
    """
    Φᵢ — intégration structurelle (formule C2TR).
    Comprehension = n_classes + 2 × n_relationships
    Récompense la richesse structurelle réelle, indépendamment du pattern déclaré.
    """
    classes = model.get("classes", [])
    rels = model.get("relationships", [])
    return float(len(classes) + 2 * len(rels))


def compute_pi(model: dict, pattern_str: str) -> float:
    """
    Π — score de conformité au pattern GoF ∈ [0, 1].

    Pour chaque pattern P détecté (ex. "Observer+Memento" → ["Observer", "Memento"]) :
        Π_methods(P) = méthodes_clés_trouvées / méthodes_clés_attendues  (schema.py)
        Π_rels(P)    = types_de_relations_trouvés / types_attendus
        Π(P)         = (Π_methods + Π_rels) / 2

    Π final = moyenne sur tous les patterns détectés.
    Φᵢ_enriched = Φᵢ × Π
    """
    from gof.schema import GOF_SCHEMA

    patterns = [p.strip() for p in pattern_str.replace("+", ",").split(",") if p.strip()]
    if not patterns:
        return 0.0

    all_methods_str = " ".join(
        m.lower()
        for cls in model.get("classes", [])
        for m in cls.get("methods", [])
    )
    rel_types_present = {r.get("type", "").lower() for r in model.get("relationships", [])}

    scores: list[float] = []
    for p in patterns:
        schema = GOF_SCHEMA.get(p)
        if schema is None:
            continue
        key_methods = schema.get("key_methods", [])
        expected_rels = schema.get("expected_rel_types", set())
        pi_m = (sum(1 for kw in key_methods if kw in all_methods_str) / len(key_methods)
                if key_methods else 1.0)
        pi_r = (len(rel_types_present & expected_rels) / len(expected_rels)
                if expected_rels else 1.0)
        scores.append((pi_m + pi_r) / 2)

    return sum(scores) / len(scores) if scores else 0.0
