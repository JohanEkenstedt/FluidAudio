import Foundation

/// A per-step hook that may re-rank the joint's top-K candidates during TDT
/// greedy decoding — e.g. to favour a list of domain terms (contextual biasing).
///
/// The decoder asks `choose` after the language filter on every step and
/// reports emitted tokens through `emitted`. Biasing a greedy decoder needs a
/// way back: when the bias picks something other than the model's top choice,
/// the decoder saves a checkpoint and, once that token is emitted, calls
/// `track()` so the bias follows that path only. If the path does not reach a
/// `completed` term, leaves every term (`matching` becomes false), or costs more
/// than `maxDeficit` logits against the model in total, the decoder rolls back
/// to the checkpoint and keeps the model's own choice at that position.
///
/// With no bias set, decoding is unchanged.
///
/// Instances are stateful and used by one decode pass at a time; see
/// `TdtTokenBiasProvider`.
public protocol TdtTokenBias: AnyObject {
    /// Called once before a decode pass.
    func reset()

    /// - Parameters:
    ///   - best: the token the decoder would emit without bias.
    ///   - blank: the blank token id.
    ///   - candidateIds: the joint's top-K token ids, best first.
    ///   - candidateLogits: their logits, in the same order.
    /// - Returns: the token to emit; `best` to keep the model's choice.
    func choose(best: Int, blank: Int, candidateIds: [Int], candidateLogits: [Float]) -> Int

    /// A non-blank token was emitted.
    func emitted(token: Int)

    /// Start following the terms the last emitted token is part of. Called
    /// right after emitting a token that overrode the model, so that a new
    /// term starting later cannot keep a dead path alive.
    func track()

    /// True while the tracked path is inside a known term.
    var matching: Bool { get }

    /// True when the last emitted token completed the tracked term.
    var completed: Bool { get }

    /// The total logit cost a biased path may have against the model.
    var maxDeficit: Float { get }

    /// Opaque state for checkpoint and rollback.
    func snapshot() -> Any?
    func restore(_ snapshot: Any?)
}

/// Creates a fresh `TdtTokenBias` for each decode pass.
///
/// Long audio is decoded in windows, possibly in parallel on worker managers,
/// so every window gets its own bias state. Expensive shared structures (the
/// term index) belong in the provider, not in each bias.
public protocol TdtTokenBiasProvider: AnyObject, Sendable {
    func makeBias() -> TdtTokenBias
}
