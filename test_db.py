import uuid
from app.database.connection import SessionLocal
from app.database.models import ResearchRun

def test_database():
    print("1. Connecting to the database...")
    db = SessionLocal()
    try:
        # Generate a fake run_id for testing
        test_id = str(uuid.uuid4())
        print(f"2. Creating a test research run with ID: {test_id}")

        # INSERT
        new_run = ResearchRun(
            run_id=test_id,
            query="Testing the PostgreSQL connection",
            status="pending",
            current_stage="initialized"
        )
        db.add(new_run)
        db.commit()
        print("   -> INSERT Successful! 🎉")

        # SELECT & UPDATE
        print("3. Fetching and updating the record...")
        run = db.query(ResearchRun).filter(ResearchRun.run_id == test_id).first()
        run.status = "completed"
        run.final_report = "# Test Report\nDatabase is working flawlessly!"
        db.commit()
        print("   -> UPDATE Successful! ✅")

        # SELECT to Verify
        updated_run = db.query(ResearchRun).filter(ResearchRun.run_id == test_id).first()
        print(f"4. Verifying data: Record status is now '{updated_run.status}'")

        # DELETE (Cleanup)
        print("5. Cleaning up test data...")
        db.delete(updated_run)
        db.commit()
        print("   -> DELETE Successful! 🧹")

        print("\n🚀 PHASE 5A IS OFFICIALLY COMPLETE! Database is 100% working.")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        db.close()
        print("6. Database connection closed.")

if __name__ == "__main__":
    test_database()
