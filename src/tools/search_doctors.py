"""Doctor search tool for Tabeeby Agent.

Provides semantic similarity search and hybrid metadata filtering (area, fee budget, specialty)
over the Vezeeta doctors vector collection.
"""

from typing import Any, Dict, List, Optional

try:
    from stores.llm.LLMInterface import LLMInterface
    from stores.vectordb.VectorDBInterface import VectorDBInterface
    from tools.catalog import normalize_area, normalize_specialty
except ImportError:
    from ..stores.llm.LLMInterface import LLMInterface
    from ..stores.vectordb.VectorDBInterface import VectorDBInterface
    from .catalog import normalize_area, normalize_specialty


# Compact summary fields to keep agent context window lean
SUMMARY_FIELDS = (
    "name",
    "specialty",
    "subspecialties_text",
    "address",
    "fee",
    "reviews_count",
    "waiting_time_min",
    "profile_url",
)


class DoctorTools:
    def __init__(
        self,
        embedding_client: LLMInterface,
        vectordb_client: VectorDBInterface,
        collection_name: str = "vezeeta_doctors",
    ):
        self.client_embedding = embedding_client
        self.client_vectorDB = vectordb_client
        self.collection_name = collection_name
            
    def _serialize_compact_doctor(self,
        payload: Dict[str, Any], score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Extract and format high-signal summary fields from a doctor payload.

        Excludes bulky clinical/biographical text (such as `about_doctor` or raw `symptoms_text`)
        to minimize LLM token consumption.
        """
        compact: Dict[str, Any] = {}
        for key in SUMMARY_FIELDS:
            val = payload.get(key)
            if val is not None and val != "":
                compact[key] = val

        if score is not None:
            compact["similarity_score"] = round(float(score), 4)

        return compact

    def search_doctors(self,
        query: str,
        area: Optional[str] = None,
        min_fee: Optional[int] = None,
        max_fee: Optional[int] = None,
        specialty: Optional[str] = None,
        limit: int = 5,
    ) -> Dict[str, Any]:
        """Search for doctors in Egypt matching clinical symptoms, specialty, area, and budget. The query is required.

        Args:
            query: Patient's medical complaint, symptoms, or condition (e.g., 'bleeding gums', 'chest tightness') Mandatory.
            area: Clinic location, neighborhood, or city in English or Arabic (e.g., 'Nasr City', 'Maadi', 'مدينة نصر').
            min_fee: Minimum consultation fee in EGP.
            max_fee: Maximum consultation fee budget in EGP.
            specialty: Medical specialty (e.g., 'Cardiology', 'Dentistry', 'Orthopedics'). Strongly recommended: always pass it when you can infer it.
            limit: Maximum number of matching doctors to return (default: 5, max: 10).

        Returns:
            Matching doctors with key details (name, specialty, address, fee, profile URL)
            plus the filters that were actually applied.
        """
        if not query or not str(query).strip():
            return {"doctors": [], "error": "A symptom or condition query is required."}

        if self.client_embedding is None or self.client_vectorDB is None:
            return {"doctors": [], "error": (
                "Doctor search is not configured. Do not retry; recommend the specialty instead."
            )}

        limit = max(1, min(int(limit or 5), 10))
        canon_area = normalize_area(area)
        canon_specialty = normalize_specialty(specialty)
        notes: List[str] = []
        if area and not canon_area:
            notes.append(f"Area '{area}' is not in the Vezeeta catalog; searched all areas.")
        if specialty and not canon_specialty:
            notes.append(f"Specialty '{specialty}' is unknown; searched all specialties.")
        if not specialty:
            # A query that IS a specialty name ("Endocrinology") is a filter, not a symptom.
            inferred = normalize_specialty(query)
            if inferred and len(str(query).split()) <= 4:
                canon_specialty = inferred
                notes.append(f"Interpreted the query as the specialty '{inferred}'.")

        vec_query = self.client_embedding.embed_text(text=str(query).strip())
        if not vec_query:
            return {"doctors": [], "error": (
                "Doctor search is temporarily unavailable (embedding service error). Do not retry; "
                "give the patient the specialty to look for and suggest https://www.vezeeta.com directly."
            )}

        # Try the strictest filter first, then relax one constraint at a time so
        # the patient gets the closest available doctors instead of nothing.
        attempts = [
            (canon_area, min_fee, max_fee, canon_specialty, None),
            (canon_area, min_fee, max_fee, None, "dropped the specialty filter"),
            (canon_area, None, None, None, "dropped the fee and specialty filters"),
            (None, min_fee, max_fee, canon_specialty, "searched outside the requested area"),
        ]
        seen = set()
        for a_area, a_min, a_max, a_spec, relaxed in attempts:
            key = (a_area, a_min, a_max, a_spec)
            if key in seen:
                continue
            seen.add(key)
            raw = self._query(vec_query, a_area, a_min, a_max, a_spec, limit)
            if raw:
                if relaxed:
                    notes.append(f"No exact match for all filters, so the search {relaxed}.")
                return {
                    "doctors": [self._serialize_compact_doctor(r.get("payload", {}), r.get("score")) for r in raw],
                    "filters_applied": {
                        "area": a_area, "min_fee": a_min, "max_fee": a_max, "specialty": a_spec,
                    },
                    "notes": notes,
                }

        return {"doctors": [], "filters_applied": {}, "notes": notes + ["No doctors matched."]}

    def _query(self, vector, area, min_fee, max_fee, specialty, limit) -> List[Dict[str, Any]]:
        query_filter = None
        if hasattr(self.client_vectorDB, "build_filter") and callable(self.client_vectorDB.build_filter):
            query_filter = self.client_vectorDB.build_filter(
                area=area, min_fee=min_fee, max_fee=max_fee, specialty=specialty,
            )
        return self.client_vectorDB.search_by_vector(
            collection_name=self.collection_name or "vezeeta_doctors",
            vector=vector,
            limit=limit,
            query_filter=query_filter,
        )

    def search_doctors_raw(
        self,
        query: str,
        area: Optional[str] = None,
        min_fee: Optional[int] = None,
        max_fee: Optional[int] = None,
        specialty: Optional[str] = None,
        limit: int = 5,
        collection_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search doctors and return raw vector DB points (including full payload and scores).

        Useful for debugging, analytics, and evaluation scripts.
        """
        if not query or not str(query).strip():
            return []

        if self.client_embedding is None or self.client_vectorDB is None:
            return []

        # Resolve collection name
        target_collection = collection_name or self.collection_name or "vezeeta_doctors"

        vec_query = self.client_embedding.embed_text(text=str(query).strip())
        if not vec_query:
            return []

        query_filter = None
        if hasattr(self.client_vectorDB, "build_filter") and callable(self.client_vectorDB.build_filter):
            query_filter = self.client_vectorDB.build_filter(
                area=area,
                min_fee=min_fee,
                max_fee=max_fee,
                specialty=specialty,
            )

        return self.client_vectorDB.search_by_vector(
            collection_name=target_collection,
            vector=vec_query,
            limit=limit,
            query_filter=query_filter,
        )