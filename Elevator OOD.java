public class Request {
	private int tragetFloor;
	Request (int tragetFloor) {
		this.tragetFloor = tragetFloor;
	}
	public int getTargetFloor () {
		return targetFloor;
	}
}

public class User {
	public void generateRequest (int targetFloor) {
		RequestHandler.getInstance.addRequest(new Request(targetFloor));
	}
}

public class Elevator {
	private int currentFloor;
	private int targetFloor;
	private int status;
	private static volatile Elevator instance = null;

	private Elevator () {
		this.currentFloor = currentFloor;
		this.targetFloor = targetFloor;
		this.status = 0;
	}

	public void moveToFloor (int targetFloor) {
		while (currentFloor < targetFloor) {
			moveUp();
		}
		while (currentFloor > targetFloor) {
			moveDown();
		}
		status = 0;
	}
	private void moveUp () {
		currentFloor++;
		status = 1;
	}
	private void moveDown () {
		currentFloor--;
		status = -1;
	}

	public static Elevator getInstance () {
		if (instance == null) {
			synchronized(this.class) {
				if (instance == null) {
					instance = new Elevator();
				}
			}
		}
		return instance;
	}
	public int getCurrentFloor () {
		return this.getCurrentFloor;
	}
	public int getStatus () {
		return this.status;
	}

}

public RequestHandler {
	List<Request> requests;
	private static volatile RequestHandler instance = null;
	public static RequestHandler getInstance () {
		if (instance == null) {
			Sychronized (RequestHandler.class) {
				if (instance == null) {
					instance = new RequestHandler();
				}
			}
		}
		return instance;
	}
	private RequestHandler () {
		requests = new ArrayList<>();
	}
	public void addRequest (Request req) {
		synchronized (req) {
			requests.add(req);
		}
	}

	private Request getNextRequest () {
		int currentFloor = Elevator.getInstance.getCurrentFloor;
		int shortestDistance = Integer.MAX_VALUE;
		Request next = null;
		for (Request req : requests) {
			if (Math.abs(req.targetFloor - currentFloor) < shortestDistance) {
				next = req;
			}
		}
		return next;
	}

	public void processRequest () {
		while (true) {
			Request  req = getNextRequest();
			if (req != null) {
				while (Elevator.getInstance().getStatus() != 0) {
					Elevator.getInstance().moveToFloor(req.getTargetFloor());
					request.remove(req);
				}
			}
		}
	}
}






