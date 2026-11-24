"""Generate small SAMPLE PDFs (study notes + PYQ papers) for 3 subjects so the
pipeline can run end to end. Replace these with real JUIT files later.

Run: python scripts/make_sample_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from app.config import PYQ_DIR, STUDY_DIR

# ---------------------------------------------------------------- study notes
# subject -> list of (unit title, [(topic, paragraphs)])
NOTES = {
    "DBMS": [
        ("Unit 1: Relational Model and SQL", [
            ("Relational Model", [
                "A relation is a table with rows (tuples) and columns (attributes). The degree of a relation is the number of attributes and the cardinality is the number of tuples.",
                "A super key is any set of attributes that uniquely identifies a tuple. A candidate key is a minimal super key. The primary key is the candidate key chosen by the designer. A foreign key is an attribute in one relation that refers to the primary key of another relation and enforces referential integrity.",
            ]),
            ("Relational Algebra", [
                "The basic operators of relational algebra are selection (sigma), projection (pi), union, set difference, Cartesian product and rename (rho). Natural join combines two relations on common attributes and removes duplicate columns.",
                "Division is used for queries containing 'for all', for example finding students who have enrolled in all courses.",
            ]),
        ]),
        ("Unit 2: Normalization", [
            ("Functional Dependencies", [
                "A functional dependency X -> Y holds if two tuples that agree on X also agree on Y. Armstrong's axioms are reflexivity, augmentation and transitivity. The closure of an attribute set X, written X+, is the set of all attributes functionally determined by X.",
            ]),
            ("Normal Forms", [
                "First Normal Form (1NF) requires atomic attribute values. Second Normal Form (2NF) requires 1NF and no partial dependency of a non-prime attribute on a candidate key. Third Normal Form (3NF) requires that for every non-trivial FD X -> A, either X is a super key or A is a prime attribute.",
                "Boyce-Codd Normal Form (BCNF) is stricter than 3NF: for every non-trivial functional dependency X -> Y, X must be a super key. Every BCNF relation is in 3NF, but a 3NF relation may not be in BCNF. BCNF decomposition is always lossless but may not preserve dependencies.",
                "A decomposition of R into R1 and R2 is lossless if R1 intersect R2 -> R1 or R1 intersect R2 -> R2 holds.",
            ]),
        ]),
        ("Unit 3: Transactions and Concurrency Control", [
            ("ACID Properties", [
                "A transaction must satisfy ACID: Atomicity (all or nothing), Consistency (moves database from one valid state to another), Isolation (concurrent transactions do not interfere) and Durability (committed changes survive failures).",
            ]),
            ("Serializability and Locking", [
                "A schedule is conflict serializable if its precedence graph is acyclic. Two operations conflict if they belong to different transactions, access the same data item and at least one is a write.",
                "Two-Phase Locking (2PL) has a growing phase, in which locks are acquired, and a shrinking phase, in which locks are released. 2PL guarantees conflict serializability but can lead to deadlock. Strict 2PL holds all exclusive locks until commit and prevents cascading rollbacks.",
            ]),
        ]),
    ],
    "Operating Systems": [
        ("Unit 1: Process Management and CPU Scheduling", [
            ("Processes", [
                "A process is a program in execution. Its states are new, ready, running, waiting and terminated. The Process Control Block (PCB) stores the process state, program counter, CPU registers, scheduling and memory-management information.",
            ]),
            ("CPU Scheduling", [
                "Turnaround Time = Completion Time - Arrival Time. Waiting Time = Turnaround Time - Burst Time. Response Time = First time on CPU - Arrival Time.",
                "First Come First Serve (FCFS) is non-preemptive and suffers from the convoy effect. Shortest Job First (SJF) gives the minimum average waiting time; its preemptive version is Shortest Remaining Time First (SRTF). Round Robin (RR) uses a time quantum; a very large quantum makes RR behave like FCFS.",
            ]),
        ]),
        ("Unit 2: Deadlocks", [
            ("Deadlock Conditions", [
                "A deadlock can occur only if four Coffman conditions hold simultaneously: mutual exclusion, hold and wait, no preemption and circular wait.",
            ]),
            ("Banker's Algorithm", [
                "The Banker's algorithm is a deadlock avoidance algorithm. Need = Max - Allocation. The system is in a safe state if there exists a safe sequence in which every process can obtain its remaining need from Available plus resources released by earlier processes.",
            ]),
        ]),
        ("Unit 3: Memory Management", [
            ("Paging", [
                "Paging divides logical memory into pages and physical memory into frames of the same size, removing external fragmentation. Effective Access Time with a TLB: EAT = h x (t + m) + (1 - h) x (t + 2m), where h is the TLB hit ratio, t the TLB access time and m the memory access time.",
            ]),
            ("Page Replacement", [
                "Page replacement algorithms include FIFO, Optimal (OPT) and Least Recently Used (LRU). Belady's anomaly is the situation where increasing the number of frames increases the number of page faults; it occurs in FIFO but not in stack algorithms such as LRU and OPT.",
                "Thrashing happens when a process spends more time paging than executing, usually because it has fewer frames than its working set.",
            ]),
        ]),
    ],
    "Computer Networks": [
        ("Unit 1: Network Models and Physical Layer", [
            ("OSI and TCP/IP Models", [
                "The OSI model has seven layers: physical, data link, network, transport, session, presentation and application. The TCP/IP model has four: link, internet, transport and application.",
            ]),
            ("Transmission", [
                "Shannon capacity: C = B log2(1 + SNR), where B is bandwidth in Hz. Nyquist bit rate for a noiseless channel: Bit Rate = 2 x B x log2(L), where L is the number of signal levels.",
            ]),
        ]),
        ("Unit 2: Data Link Layer", [
            ("Error Control", [
                "Cyclic Redundancy Check (CRC) appends the remainder of modulo-2 division by a generator polynomial. Hamming code can correct single-bit errors; the number of redundant bits r satisfies 2^r >= m + r + 1.",
            ]),
            ("Medium Access Control", [
                "In Pure ALOHA the maximum throughput is 18.4 percent (S = G e^(-2G)); in Slotted ALOHA it is 36.8 percent (S = G e^(-G)). CSMA/CD is used in classic Ethernet; the minimum frame size must satisfy Transmission time >= 2 x Propagation delay.",
                "Stop-and-wait efficiency = 1 / (1 + 2a), where a = Propagation delay / Transmission delay. Sliding window with window size W has efficiency W / (1 + 2a) when W < 1 + 2a.",
            ]),
        ]),
        ("Unit 3: Network and Transport Layer", [
            ("IP Addressing", [
                "An IPv4 address is 32 bits. In CIDR notation /n, the number of host addresses is 2^(32 - n) - 2. Subnetting borrows host bits to create smaller networks.",
            ]),
            ("TCP", [
                "TCP uses a three-way handshake (SYN, SYN-ACK, ACK) to establish a connection. TCP congestion control uses slow start, where the congestion window doubles every RTT until the threshold, then congestion avoidance, where it grows by one MSS per RTT.",
            ]),
        ]),
    ],
}

# ---------------------------------------------------------------- PYQ papers
# subject -> {year: [(question, marks, unit)]}
PYQS = {
    "DBMS": {
        2023: [
            ("Define super key, candidate key and primary key with an example.", 3, 1),
            ("Explain the division operator of relational algebra with a suitable query.", 5, 1),
            ("Differentiate between 3NF and BCNF. Is every BCNF relation in 3NF? Justify.", 5, 2),
            ("Explain the ACID properties of a transaction.", 4, 3),
            ("What is two-phase locking? Show that it can lead to deadlock.", 5, 3),
        ],
        2024: [
            ("Given R(A,B,C,D) with FDs A->B, B->C, find the closure of A and all candidate keys.", 5, 2),
            ("When is a decomposition lossless? Give the condition.", 3, 2),
            ("Check whether a schedule is conflict serializable using a precedence graph.", 5, 3),
            ("What is the role of a foreign key in referential integrity?", 2, 1),
        ],
    },
    "Operating Systems": {
        2023: [
            ("Draw the process state diagram and explain each state.", 4, 1),
            ("Compute average waiting time for FCFS and SJF for the given processes.", 6, 1),
            ("State the four necessary conditions for deadlock.", 3, 2),
            ("Explain Belady's anomaly with an example reference string.", 5, 3),
        ],
        2024: [
            ("Apply the Banker's algorithm to check whether the system is in a safe state.", 6, 2),
            ("Derive the effective access time for paging with a TLB.", 4, 3),
            ("What is thrashing and how can it be prevented?", 3, 3),
            ("Explain Round Robin scheduling. What happens if the time quantum is very large?", 4, 1),
        ],
    },
    "Computer Networks": {
        2023: [
            ("Compare the OSI and TCP/IP reference models.", 4, 1),
            ("Calculate the channel capacity using Shannon's theorem for B = 3000 Hz and SNR = 3162.", 4, 1),
            ("Derive the maximum throughput of Pure ALOHA.", 5, 2),
            ("Explain the TCP three-way handshake.", 3, 3),
        ],
        2024: [
            ("Find the efficiency of stop-and-wait if propagation delay is 20 ms and transmission time is 10 ms.", 4, 2),
            ("How many redundant bits are needed for Hamming code with 7 data bits?", 3, 2),
            ("Given 192.168.10.0/26, find the number of usable hosts per subnet.", 4, 3),
            ("Explain slow start and congestion avoidance in TCP.", 5, 3),
        ],
    },
}


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def build_notes(subject: str, units, path: Path):
    styles = getSampleStyleSheet()
    story = [Paragraph(f"{subject} - Study Notes (SAMPLE)", styles["Title"])]
    for i, (unit, topics) in enumerate(units):
        if i:
            story.append(PageBreak())
        story.append(Paragraph(unit, styles["Heading1"]))
        for topic, paras in topics:
            story.append(Paragraph(f"Topic: {topic}", styles["Heading2"]))
            for p in paras:
                story.append(Paragraph(p, styles["BodyText"]))
                story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def build_pyq(subject: str, year: int, questions, path: Path):
    styles = getSampleStyleSheet()
    story = [
        Paragraph(f"SAMPLE End Term Examination {year}", styles["Title"]),
        Paragraph(f"Subject: {subject}", styles["Heading2"]),
        Paragraph(f"Year: {year}    Max Marks: 35", styles["BodyText"]),
        Spacer(1, 10),
    ]
    for n, (q, marks, unit) in enumerate(questions, 1):
        story.append(Paragraph(f"Q{n}. {q} [{marks} marks] (Unit {unit})", styles["BodyText"]))
        story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def main():
    for subject, units in NOTES.items():
        d = STUDY_DIR / slug(subject)
        d.mkdir(parents=True, exist_ok=True)
        build_notes(subject, units, d / f"{slug(subject)}_notes.pdf")
    for subject, years in PYQS.items():
        d = PYQ_DIR / slug(subject)
        d.mkdir(parents=True, exist_ok=True)
        for year, qs in years.items():
            build_pyq(subject, year, qs, d / f"{slug(subject)}_endterm_{year}.pdf")
    for p in sorted(STUDY_DIR.rglob("*.pdf")) + sorted(PYQ_DIR.rglob("*.pdf")):
        print("wrote", p.relative_to(p.parents[3]))


if __name__ == "__main__":
    main()
