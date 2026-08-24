import { getMetrics } from "@/lib/api-client";

// This page's data lives entirely on a separately-deployed backend, so it
// can't be prerendered at frontend build time -- render it per-request.
export const dynamic = "force-dynamic";

function pct(value: number | undefined): string {
  if (value === undefined) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

export default async function AboutPage() {
  const metrics = await getMetrics();

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-semibold">About this model</h1>
        <div className="mt-3 flex flex-col gap-3 text-sm" style={{ color: "var(--muted)" }}>
          <p>
            Trained offline on <strong>goodbooks-10k</strong> (~6M ratings, 10K books, 53K
            users). Models progress from a popularity baseline through classic collaborative
            filtering (item-kNN, SVD, ALS) to a content-embedding model
            (sentence-transformers) and two deep-learning models: Neural Collaborative
            Filtering and a two-tower retrieval model. The deployed model is a hybrid that
            blends the two-tower score with content similarity, falling back to pure content
            similarity for books with too few ratings to trust collaborative signal.
          </p>
          <p>
            goodbooks-10k has no rating timestamps, so evaluation uses a leave-one-out split
            per user rather than a time-based split. The cold-start columns below isolate
            users/books with fewer than 5 training interactions — this is where the hybrid
            model is expected to clearly outperform pure collaborative filtering.
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left" style={{ borderColor: "var(--border)" }}>
              <th className="py-2 pr-4">Model</th>
              <th className="py-2 pr-4">Precision@10</th>
              <th className="py-2 pr-4">Recall@10</th>
              <th className="py-2 pr-4">NDCG@10</th>
              <th className="py-2 pr-4">Hit Rate@10</th>
              <th className="py-2 pr-4">Coverage@10</th>
              <th className="py-2 pr-4">Cold-user NDCG@10</th>
              <th className="py-2 pr-4">Cold-item Recall@10</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((row) => (
              <tr key={row.model} className="border-b" style={{ borderColor: "var(--border)" }}>
                <td className="py-2 pr-4 font-medium">{row.model}</td>
                <td className="py-2 pr-4">{pct(row.overall?.precision?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.overall?.recall?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.overall?.ndcg?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.overall?.hit_rate?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.overall?.coverage?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.cold_start?.cold_user_metrics?.ndcg?.["10"])}</td>
                <td className="py-2 pr-4">{pct(row.cold_start?.cold_item_recall?.["10"])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
